// bf16_alu.v — Single-lane BFloat16 ALU (2-cycle pipelined)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module bf16_alu (
    input                 clk,
    input                 rst,
    input  [3:0]          alu_op,
    input  [15:0]         a,
    input  [15:0]         b,
    output reg [15:0]     result
);

    //--- EX1 ---

    // Field extraction
    wire        a_sign = a[15];
    wire [7:0]  a_exp  = a[14:7];
    wire [6:0]  a_mant = a[6:0];
    wire        b_sign = b[15];
    wire [7:0]  b_exp  = b[14:7];
    wire [6:0]  b_mant = b[6:0];

    // Special value flags
    wire a_is_zero   = (a_exp == 8'd0) && (a_mant == 7'd0);
    wire b_is_zero   = (b_exp == 8'd0) && (b_mant == 7'd0);
    wire a_is_inf    = (a_exp == 8'hFF) && (a_mant == 7'd0);
    wire b_is_inf    = (b_exp == 8'hFF) && (b_mant == 7'd0);
    wire a_is_nan    = (a_exp == 8'hFF) && (a_mant != 7'd0);
    wire b_is_nan    = (b_exp == 8'hFF) && (b_mant != 7'd0);
    wire a_is_denorm = (a_exp == 8'd0) && (a_mant != 7'd0);
    wire b_is_denorm = (b_exp == 8'd0) && (b_mant != 7'd0);

    // MUL
    wire        mul_sign = a_sign ^ b_sign;
    wire [7:0]  a_full = {1'b1, a_mant};
    wire [7:0]  b_full = {1'b1, b_mant};
    wire [15:0] mul_product = a_full * b_full;

    wire        mul_norm_shift = mul_product[15];
    wire [6:0]  mul_norm_mant = mul_norm_shift ? mul_product[14:8]
                                               : mul_product[13:7];

    // Unsigned exponent arithmetic (avoids ISE 10.1 signed bug)
    wire [8:0] mul_ab_sum    = {1'b0, a_exp} + {1'b0, b_exp};
    wire [9:0] mul_exp_n0    = {1'b0, mul_ab_sum}     - 10'd127;
    wire [9:0] mul_exp_n1    = {1'b0, mul_ab_sum} + 10'd1 - 10'd127;
    wire [9:0] mul_norm_exp  = mul_norm_shift ? mul_exp_n1 : mul_exp_n0;

    wire mul_exp_under = mul_norm_exp[9] || (mul_norm_exp == 10'd0);
    wire mul_exp_over  = !mul_norm_exp[9] && (mul_norm_exp >= 10'd255);

    wire [15:0] mul_result =
        (a_is_nan || b_is_nan)                                      ? `BF16_NAN :
        ((a_is_inf && (b_is_zero || b_is_denorm)) ||
         (b_is_inf && (a_is_zero || a_is_denorm)))                  ? `BF16_NAN :
        (a_is_inf || b_is_inf)                                      ? {mul_sign, 8'hFF, 7'd0} :
        (a_is_zero || b_is_zero || a_is_denorm || b_is_denorm)      ? {mul_sign, 15'd0} :
        mul_exp_under                                                ? {mul_sign, 15'd0} :
        mul_exp_over                                                 ? {mul_sign, 8'hFF, 7'd0} :
        {mul_sign, mul_norm_exp[7:0], mul_norm_mant};

    // ADD/SUB
    wire add_b_sign = (alu_op == `ALU_SUB) ? ~b_sign : b_sign;

    wire a_gt_b_mag = (a_exp > b_exp) ||
                      ((a_exp == b_exp) && (a_mant >= b_mant));

    wire [7:0]  large_exp  = a_gt_b_mag ? a_exp      : b_exp;
    wire [6:0]  large_mant = a_gt_b_mag ? a_mant     : b_mant;
    wire        large_sign = a_gt_b_mag ? a_sign      : add_b_sign;
    wire [7:0]  small_exp  = a_gt_b_mag ? b_exp       : a_exp;
    wire [6:0]  small_mant = a_gt_b_mag ? b_mant      : a_mant;
    wire        small_sign = a_gt_b_mag ? add_b_sign   : a_sign;

    wire large_is_zero = (large_exp == 8'd0);
    wire small_is_zero = (small_exp == 8'd0);

    wire [8:0] large_full = large_is_zero ? 9'd0 : {1'b1, large_mant, 1'b0};
    wire [8:0] small_full_pre = small_is_zero ? 9'd0 : {1'b1, small_mant, 1'b0};

    wire [7:0] exp_diff = large_exp - small_exp;
    wire [3:0] shift_amt = (exp_diff > 8'd9) ? 4'd9 : exp_diff[3:0];
    wire [8:0] small_full = small_full_pre >> shift_amt;

    wire eff_sub = (large_sign != small_sign);

    wire [9:0] add_raw = eff_sub ? ({1'b0, large_full} - {1'b0, small_full})
                                 : ({1'b0, large_full} + {1'b0, small_full});

    wire add_res_sign = large_sign;

    // MAX
    wire a_gt_b_val =
        (a_is_nan)                      ? 1'b0 :
        (b_is_nan)                      ? 1'b1 :
        (a_sign != b_sign)              ? b_sign :
        (a_sign == 1'b0)                ? a_gt_b_mag :
                                          !a_gt_b_mag;

    wire [15:0] max_result = (a_is_nan || b_is_nan) ? `BF16_NAN :
                             a_gt_b_val ? a : b;

    // MOV / RELU
    wire [15:0] mov_result  = a;
    wire [15:0] relu_result = a[15] ? `BF16_ZERO_POS : a;

    // ADD/SUB special-case flags for EX2
    wire any_nan_add    = a_is_nan || b_is_nan;
    wire inf_sub_inf    = a_is_inf && b_is_inf && (a_sign != add_b_sign);
    wire a_eff_zero     = a_is_zero || a_is_denorm;
    wire add_b_is_zero  = b_is_zero || b_is_denorm;

    //--- Pipeline Register: EX1 -> EX2 ---

    reg [3:0]   s1_alu_op;
    reg [9:0]   s1_add_raw;
    reg [7:0]   s1_large_exp;
    reg         s1_eff_sub;
    reg         s1_add_res_sign;
    reg         s1_any_nan;
    reg         s1_inf_sub_inf;
    reg         s1_a_is_inf;
    reg         s1_b_is_inf;
    reg         s1_a_eff_zero;
    reg         s1_add_b_is_zero;
    reg [15:0]  s1_a;
    reg         s1_add_b_sign;
    reg [14:0]  s1_b_bits;
    reg [15:0]  s1_mul_result;
    reg [15:0]  s1_max_result;
    reg [15:0]  s1_mov_result;
    reg [15:0]  s1_relu_result;

    always @(posedge clk) begin
        if (rst) begin
            s1_alu_op        <= 4'd0;
            s1_add_raw       <= 10'd0;
            s1_large_exp     <= 8'd0;
            s1_eff_sub       <= 1'b0;
            s1_add_res_sign  <= 1'b0;
            s1_any_nan       <= 1'b0;
            s1_inf_sub_inf   <= 1'b0;
            s1_a_is_inf      <= 1'b0;
            s1_b_is_inf      <= 1'b0;
            s1_a_eff_zero    <= 1'b0;
            s1_add_b_is_zero <= 1'b0;
            s1_a             <= 16'd0;
            s1_add_b_sign    <= 1'b0;
            s1_b_bits        <= 15'd0;
            s1_mul_result    <= 16'd0;
            s1_max_result    <= 16'd0;
            s1_mov_result    <= 16'd0;
            s1_relu_result   <= 16'd0;
        end else begin
            s1_alu_op        <= alu_op;
            s1_add_raw       <= add_raw;
            s1_large_exp     <= large_exp;
            s1_eff_sub       <= eff_sub;
            s1_add_res_sign  <= add_res_sign;
            s1_any_nan       <= any_nan_add;
            s1_inf_sub_inf   <= inf_sub_inf;
            s1_a_is_inf      <= a_is_inf;
            s1_b_is_inf      <= b_is_inf;
            s1_a_eff_zero    <= a_eff_zero;
            s1_add_b_is_zero <= add_b_is_zero;
            s1_a             <= a;
            s1_add_b_sign    <= add_b_sign;
            s1_b_bits        <= b[14:0];
            s1_mul_result    <= mul_result;
            s1_max_result    <= max_result;
            s1_mov_result    <= mov_result;
            s1_relu_result   <= relu_result;
        end
    end

    //--- EX2: LZC + normalization + output mux ---

    // Leading zero count
    reg [3:0] lzc;
    always @(*) begin
        casez (s1_add_raw[8:0])
            9'b1????????: lzc = 4'd0;
            9'b01???????: lzc = 4'd1;
            9'b001??????: lzc = 4'd2;
            9'b0001?????: lzc = 4'd3;
            9'b00001????: lzc = 4'd4;
            9'b000001???: lzc = 4'd5;
            9'b0000001??: lzc = 4'd6;
            9'b00000001?: lzc = 4'd7;
            9'b000000001: lzc = 4'd8;
            default:      lzc = 4'd9;
        endcase
    end

    // Normalization
    wire add_overflow = s1_add_raw[9] && !s1_eff_sub;

    wire [8:0] add_shifted = add_overflow ? s1_add_raw[9:1]
                                          : (s1_add_raw[8:0] << lzc);

    // Unsigned exponent adjustment (avoids ISE 10.1 signed bug)
    wire [9:0] add_exp_adj = add_overflow
        ? ({2'b0, s1_large_exp} + 10'd1)
        : ({2'b0, s1_large_exp} - {6'd0, lzc});
    wire add_exp_underflow = !add_overflow && ({6'd0, lzc} > {2'b0, s1_large_exp});

    wire [6:0] add_final_mant = add_shifted[7:1];
    wire [7:0] add_final_exp  = add_exp_adj[7:0];

    // Split ADD/SUB into special and normal paths
    wire use_special = s1_any_nan || s1_inf_sub_inf ||
                       s1_a_is_inf || s1_b_is_inf ||
                       (s1_a_eff_zero && s1_add_b_is_zero) ||
                       s1_a_eff_zero || s1_add_b_is_zero ||
                       (s1_add_raw == 10'd0);

    wire [15:0] special_result =
        (s1_any_nan || s1_inf_sub_inf)      ? `BF16_NAN :
        s1_a_is_inf                         ? s1_a :
        s1_b_is_inf                         ? {s1_add_b_sign, s1_b_bits} :
        (s1_a_eff_zero && s1_add_b_is_zero) ? `BF16_ZERO_POS :
        s1_a_eff_zero                       ? {s1_add_b_sign, s1_b_bits} :
        s1_add_b_is_zero                    ? s1_a :
        `BF16_ZERO_POS;

    wire [15:0] normal_result =
        (add_exp_underflow || add_exp_adj == 10'd0 || add_exp_adj[9]) ? `BF16_ZERO_POS :
        (!add_exp_adj[9] && add_exp_adj >= 10'd255)                   ? {s1_add_res_sign, 8'hFF, 7'd0} :
        {s1_add_res_sign, add_final_exp, add_final_mant};

    wire [15:0] add_result = use_special ? special_result : normal_result;

    // Output MUX
    always @(*) begin
        case (s1_alu_op)
            `ALU_ADD:  result = add_result;
            `ALU_SUB:  result = add_result;
            `ALU_MUL:  result = s1_mul_result;
            `ALU_MAX:  result = s1_max_result;
            `ALU_MOV:  result = s1_mov_result;
            `ALU_BCAST: result = s1_mov_result;
            `ALU_RELU: result = s1_relu_result;
            default:   result = `BF16_ZERO_POS;
        endcase
    end

endmodule
