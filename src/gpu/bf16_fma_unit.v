// bf16_fma_unit.v — BF16 fused multiply-add (3-cycle pipeline)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module bf16_fma_unit (
    input         clk,
    input         rst,
    input  [15:0] a,
    input  [15:0] b,
    input  [15:0] c,
    output reg [15:0] result
);

    //--- CYCLE 1: Multiply + exponent arithmetic + magnitude sort ---

    // Field extraction
    wire        a_sign = a[15];
    wire [7:0]  a_exp  = a[14:7];
    wire [6:0]  a_mant = a[6:0];
    wire        b_sign = b[15];
    wire [7:0]  b_exp  = b[14:7];
    wire [6:0]  b_mant = b[6:0];
    wire        c_sign = c[15];
    wire [7:0]  c_exp  = c[14:7];
    wire [6:0]  c_mant = c[6:0];

    // Special value flags
    wire a_is_zero   = (a_exp == 8'd0) && (a_mant == 7'd0);
    wire b_is_zero   = (b_exp == 8'd0) && (b_mant == 7'd0);
    wire c_is_zero   = (c_exp == 8'd0) && (c_mant == 7'd0);
    wire a_is_inf    = (a_exp == 8'hFF) && (a_mant == 7'd0);
    wire b_is_inf    = (b_exp == 8'hFF) && (b_mant == 7'd0);
    wire c_is_inf    = (c_exp == 8'hFF) && (c_mant == 7'd0);
    wire a_is_nan    = (a_exp == 8'hFF) && (a_mant != 7'd0);
    wire b_is_nan    = (b_exp == 8'hFF) && (b_mant != 7'd0);
    wire c_is_nan    = (c_exp == 8'hFF) && (c_mant != 7'd0);
    wire a_is_denorm = (a_exp == 8'd0) && (a_mant != 7'd0);
    wire b_is_denorm = (b_exp == 8'd0) && (b_mant != 7'd0);
    wire c_is_denorm = (c_exp == 8'd0) && (c_mant != 7'd0);

    wire ab_is_zero = a_is_zero || b_is_zero || a_is_denorm || b_is_denorm;
    wire c_eff_zero = c_is_zero || c_is_denorm;

    // A * B mantissa multiply (infers MULT18X18)
    wire        mul_sign = a_sign ^ b_sign;
    wire [7:0]  a_full = {1'b1, a_mant};
    wire [7:0]  b_full = {1'b1, b_mant};
    wire [15:0] product = a_full * b_full;

    wire [17:0] c_mant_wide = {1'b1, c_mant, 10'b0};

    // Pre-compute exponent for both mul_norm cases (unsigned)
    wire [8:0] ab_exp_sum = {1'b0, a_exp} + {1'b0, b_exp};
    wire [8:0] ab_sum_c0  = ab_exp_sum;
    wire [8:0] ab_sum_c1  = ab_exp_sum + 9'd1;
    wire [8:0] c_exp_bias = 9'd127 + {1'b0, c_exp};

    wire c0_ge = (ab_sum_c0 >= c_exp_bias);
    wire c1_ge = (ab_sum_c1 >= c_exp_bias);

    wire [8:0] diff_c0_fwd = ab_sum_c0 - c_exp_bias;
    wire [8:0] diff_c0_rev = c_exp_bias - ab_sum_c0;
    wire [8:0] diff_c1_fwd = ab_sum_c1 - c_exp_bias;
    wire [8:0] diff_c1_rev = c_exp_bias - ab_sum_c1;

    wire [8:0] abs_diff_c0 = c0_ge ? diff_c0_fwd : diff_c0_rev;
    wire [8:0] abs_diff_c1 = c1_ge ? diff_c1_fwd : diff_c1_rev;

    wire [4:0] sa_c0 = (abs_diff_c0 > 9'd18) ? 5'd18 : abs_diff_c0[4:0];
    wire [4:0] sa_c1 = (abs_diff_c1 > 9'd18) ? 5'd18 : abs_diff_c1[4:0];

    wire gt_c0 = c0_ge && (ab_sum_c0 != c_exp_bias);
    wire gt_c1 = c1_ge && (ab_sum_c1 != c_exp_bias);
    wire eq_c0 = (ab_sum_c0 == c_exp_bias);
    wire eq_c1 = (ab_sum_c1 == c_exp_bias);

    // Post-multiply selection based on product[15]
    wire mul_norm = product[15];

    wire [9:0] mul_exp_c0 = {1'b0, ab_sum_c0} - 10'd127;
    wire [9:0] mul_exp_c1 = {1'b0, ab_sum_c1} - 10'd127;
    wire [9:0] mul_exp_norm = mul_norm ? mul_exp_c1 : mul_exp_c0;

    wire [17:0] mul_mant_wide = mul_norm ? {1'b1, product[14:0], 2'b0}
                                         : {1'b1, product[13:0], 3'b0};

    wire exp_gt      = mul_norm ? gt_c1 : gt_c0;
    wire exp_eq      = mul_norm ? eq_c1 : eq_c0;
    wire exp_ge      = mul_norm ? c1_ge : c0_ge;
    wire [4:0] shift_amt = mul_norm ? sa_c1 : sa_c0;

    wire mul_mag_ge = exp_gt || (exp_eq && (mul_mant_wide >= c_mant_wide));

    // Sort by magnitude
    wire [17:0] large_mant = mul_mag_ge ? mul_mant_wide : c_mant_wide;
    wire [17:0] small_mant = mul_mag_ge ? c_mant_wide   : mul_mant_wide;
    wire [9:0] aligned_exp = mul_mag_ge ? mul_exp_norm : {2'b0, c_exp};

    wire eff_sub  = (mul_sign != c_sign);
    wire res_sign = eff_sub ? (mul_mag_ge ? mul_sign : c_sign) : mul_sign;

    // Standalone mul result (for c_eff_zero fast path)
    wire mul_exp_under = mul_exp_norm[9] || (mul_exp_norm == 10'd0);
    wire mul_exp_over  = !mul_exp_norm[9] && (mul_exp_norm >= 10'd255);
    wire [6:0] mul_only_mant = mul_mant_wide[16:10];
    wire [15:0] mul_only_result =
        mul_exp_under ? {mul_sign, 15'd0} :
        mul_exp_over  ? {mul_sign, 8'hFF, 7'd0} :
        {mul_sign, mul_exp_norm[7:0], mul_only_mant};

    wire any_nan = a_is_nan || b_is_nan || c_is_nan;
    wire mul_inf = a_is_inf || b_is_inf;
    wire inf_times_zero = (a_is_inf && (b_is_zero || b_is_denorm)) ||
                          (b_is_inf && (a_is_zero || a_is_denorm));

    //--- Pipeline Register A: Cycle 1 -> Cycle 2 ---

    reg [17:0] s1_large_mant;
    reg [17:0] s1_small_mant;
    reg [4:0]  s1_shift_amt;
    reg [9:0]  s1_aligned_exp;
    reg        s1_eff_sub;
    reg        s1_res_sign;
    reg        s1_mul_sign;
    reg        s1_c_sign;
    reg        s1_ab_is_zero;
    reg        s1_c_eff_zero;
    reg        s1_any_nan;
    reg        s1_mul_inf;
    reg        s1_inf_times_zero;
    reg        s1_c_is_inf;
    reg [15:0] s1_c;
    reg [15:0] s1_mul_only_result;

    always @(posedge clk) begin
        if (rst) begin
            s1_large_mant      <= 18'd0;
            s1_small_mant      <= 18'd0;
            s1_shift_amt       <= 5'd0;
            s1_aligned_exp     <= 10'd0;
            s1_eff_sub         <= 1'b0;
            s1_res_sign        <= 1'b0;
            s1_mul_sign        <= 1'b0;
            s1_c_sign          <= 1'b0;
            s1_ab_is_zero      <= 1'b0;
            s1_c_eff_zero      <= 1'b0;
            s1_any_nan         <= 1'b0;
            s1_mul_inf         <= 1'b0;
            s1_inf_times_zero  <= 1'b0;
            s1_c_is_inf        <= 1'b0;
            s1_c               <= 16'd0;
            s1_mul_only_result <= 16'd0;
        end else begin
            s1_large_mant      <= large_mant;
            s1_small_mant      <= small_mant;
            s1_shift_amt       <= shift_amt;
            s1_aligned_exp     <= aligned_exp;
            s1_eff_sub         <= eff_sub;
            s1_res_sign        <= res_sign;
            s1_mul_sign        <= mul_sign;
            s1_c_sign          <= c_sign;
            s1_ab_is_zero      <= ab_is_zero;
            s1_c_eff_zero      <= c_eff_zero;
            s1_any_nan         <= any_nan;
            s1_mul_inf         <= mul_inf;
            s1_inf_times_zero  <= inf_times_zero;
            s1_c_is_inf        <= c_is_inf;
            s1_c               <= c;
            s1_mul_only_result <= mul_only_result;
        end
    end

    //--- CYCLE 2: Barrel shift + add/sub ---

    wire [17:0] small_shifted = s1_small_mant >> s1_shift_amt;

    wire [18:0] sum = s1_eff_sub ? ({1'b0, s1_large_mant} - {1'b0, small_shifted})
                                 : ({1'b0, s1_large_mant} + {1'b0, small_shifted});

    //--- Pipeline Register B: Cycle 2 -> Cycle 3 ---

    reg [18:0] s2_sum;
    reg [9:0]  s2_aligned_exp;
    reg        s2_eff_sub;
    reg        s2_res_sign;
    reg        s2_mul_sign;
    reg        s2_c_sign;
    reg        s2_ab_is_zero;
    reg        s2_c_eff_zero;
    reg        s2_any_nan;
    reg        s2_mul_inf;
    reg        s2_inf_times_zero;
    reg        s2_c_is_inf;
    reg [15:0] s2_c;
    reg [15:0] s2_mul_only_result;

    always @(posedge clk) begin
        if (rst) begin
            s2_sum             <= 19'd0;
            s2_aligned_exp     <= 10'd0;
            s2_eff_sub         <= 1'b0;
            s2_res_sign        <= 1'b0;
            s2_mul_sign        <= 1'b0;
            s2_c_sign          <= 1'b0;
            s2_ab_is_zero      <= 1'b0;
            s2_c_eff_zero      <= 1'b0;
            s2_any_nan         <= 1'b0;
            s2_mul_inf         <= 1'b0;
            s2_inf_times_zero  <= 1'b0;
            s2_c_is_inf        <= 1'b0;
            s2_c               <= 16'd0;
            s2_mul_only_result <= 16'd0;
        end else begin
            s2_sum             <= sum;
            s2_aligned_exp     <= s1_aligned_exp;
            s2_eff_sub         <= s1_eff_sub;
            s2_res_sign        <= s1_res_sign;
            s2_mul_sign        <= s1_mul_sign;
            s2_c_sign          <= s1_c_sign;
            s2_ab_is_zero      <= s1_ab_is_zero;
            s2_c_eff_zero      <= s1_c_eff_zero;
            s2_any_nan         <= s1_any_nan;
            s2_mul_inf         <= s1_mul_inf;
            s2_inf_times_zero  <= s1_inf_times_zero;
            s2_c_is_inf        <= s1_c_is_inf;
            s2_c               <= s1_c;
            s2_mul_only_result <= s1_mul_only_result;
        end
    end

    //--- CYCLE 3: LZC + normalize + output mux ---

    wire add_overflow = s2_sum[18] && !s2_eff_sub;

    // 18-bit leading zero count
    reg [4:0] lzc;
    always @(*) begin
        casez (s2_sum[17:0])
            18'b1?????????????????: lzc = 5'd0;
            18'b01????????????????: lzc = 5'd1;
            18'b001???????????????: lzc = 5'd2;
            18'b0001??????????????: lzc = 5'd3;
            18'b00001?????????????: lzc = 5'd4;
            18'b000001????????????: lzc = 5'd5;
            18'b0000001???????????: lzc = 5'd6;
            18'b00000001??????????: lzc = 5'd7;
            18'b000000001?????????: lzc = 5'd8;
            18'b0000000001????????: lzc = 5'd9;
            18'b00000000001???????: lzc = 5'd10;
            18'b000000000001??????: lzc = 5'd11;
            18'b0000000000001?????: lzc = 5'd12;
            18'b00000000000001????: lzc = 5'd13;
            18'b000000000000001???: lzc = 5'd14;
            18'b0000000000000001??: lzc = 5'd15;
            18'b00000000000000001?: lzc = 5'd16;
            18'b000000000000000001: lzc = 5'd17;
            default:                lzc = 5'd18;
        endcase
    end

    // Normalization
    wire [17:0] norm_mant = add_overflow ? s2_sum[18:1]
                                         : (s2_sum[17:0] << lzc);

    // Unsigned exponent adjustment
    wire [9:0] exp_after_lzc = s2_aligned_exp - {5'b0, lzc};
    wire exp_lzc_underflow = ({5'b0, lzc} > s2_aligned_exp);

    wire [9:0] exp_after_ovf = s2_aligned_exp + 10'd1;

    wire [9:0] norm_exp = add_overflow ? exp_after_ovf : exp_after_lzc;
    wire norm_exp_neg = add_overflow ? 1'b0 : exp_lzc_underflow;

    wire [6:0] final_mant = norm_mant[16:10];
    wire [7:0] final_exp  = norm_exp[7:0];

    wire norm_under = norm_exp_neg || (norm_exp == 10'd0) || s2_aligned_exp[9];
    wire norm_over  = !norm_exp_neg && !s2_aligned_exp[9] && (norm_exp >= 10'd255);

    wire [15:0] fma_normal =
        (s2_sum == 19'd0)  ? `BF16_ZERO_POS :
        norm_under          ? `BF16_ZERO_POS :
        norm_over           ? {s2_res_sign, 8'hFF, 7'd0} :
        {s2_res_sign, final_exp, final_mant};

    // Special case priority + output MUX
    always @(*) begin
        if (s2_any_nan || s2_inf_times_zero)
            result = `BF16_NAN;
        else if (s2_mul_inf && s2_c_is_inf && (s2_mul_sign != s2_c_sign))
            result = `BF16_NAN;
        else if (s2_mul_inf)
            result = {s2_mul_sign, 8'hFF, 7'd0};
        else if (s2_c_is_inf)
            result = s2_c;
        else if (s2_ab_is_zero && s2_c_eff_zero)
            result = `BF16_ZERO_POS;
        else if (s2_ab_is_zero)
            result = s2_c;
        else if (s2_c_eff_zero)
            result = s2_mul_only_result;
        else
            result = fma_normal;
    end

endmodule
