// simd_alu.v — 4-lane SIMD ALU wrapper (3-cycle pipelined)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module simd_alu (
    input                       clk,
    input                       rst,
    input  [3:0]                alu_op,
    input  [2:0]                dtype,
    input  [`DATA_WIDTH-1:0]    operand_a,
    input  [`DATA_WIDTH-1:0]    operand_b,
    output [`DATA_WIDTH-1:0]    result
);

    wire [15:0] bf16_lane_result [0:`SIMD_LANES-1];
    genvar j;

    // BCAST: select one lane from operand_a and replicate to all 4 positions
    wire [1:0] bcast_lane_sel = operand_b[1:0];
    reg [15:0] bcast_value;
    always @(*) begin
        case (bcast_lane_sel)
            2'd0: bcast_value = operand_a[15:0];
            2'd1: bcast_value = operand_a[31:16];
            2'd2: bcast_value = operand_a[47:32];
            default: bcast_value = operand_a[63:48];
        endcase
    end

    wire [`DATA_WIDTH-1:0] effective_a = (alu_op == `ALU_BCAST)
        ? {bcast_value, bcast_value, bcast_value, bcast_value}
        : operand_a;

    // EX1 -> EX2 pipeline register (fix 1 / milestone 3).
    // Breaks the critical path: use_imm -> op_b mux -> bcast_value mux ->
    // effective_a -> bf16.a_gt_b_mag -> exp_diff -> shift -> add_raw -> FF
    // was 15.26 ns (twr Apr 17). Registering BCAST mux output + ALU controls
    // splits it into ~6 ns (input -> this reg) + ~7 ns (reg -> bf16 s1_* FF).
    // Total simd_alu latency grows from 2 to 3 cycles; execute_stage drops
    // its own simd_result_r so external SIMD latency is unchanged.
    reg [3:0]              alu_op_r;
    reg [2:0]              dtype_r1;
    reg [`DATA_WIDTH-1:0]  effective_a_r;
    reg [`DATA_WIDTH-1:0]  operand_b_r;

    always @(posedge clk) begin
        if (rst) begin
            alu_op_r      <= 4'd0;
            dtype_r1      <= 3'd0;
            effective_a_r <= {`DATA_WIDTH{1'b0}};
            operand_b_r   <= {`DATA_WIDTH{1'b0}};
        end else begin
            alu_op_r      <= alu_op;
            dtype_r1      <= dtype;
            effective_a_r <= effective_a;
            operand_b_r   <= operand_b;
        end
    end

    // Scalar INT16 ALU (lane 0 only, replicated to all 4 lanes).
    // Reduced from 4-lane SIMD — IDS / ANN kernels only consume lane 0 of
    // integer results (pointer arithmetic, loop counters). Frees 3 MULT18X18
    // + ~250 LUTs. See docs/milestone3_refinements.md.
    wire [15:0] int_lane0;
    int16_alu u_alu (
        .alu_op (alu_op_r),
        .a      (effective_a_r[15:0]),
        .b      (operand_b_r[15:0]),
        .result (int_lane0)
    );

    // BF16 ALU instances (2-cycle internal; with input reg, output appears
    // 3 cycles after simd_alu inputs arrive)
    generate
        for (j = 0; j < `SIMD_LANES; j = j + 1) begin : bf16_lane
            bf16_alu u_bf16 (
                .clk    (clk),
                .rst    (rst),
                .alu_op (alu_op_r),
                .a      (effective_a_r[j*16+15 : j*16]),
                .b      (operand_b_r[j*16+15 : j*16]),
                .result (bf16_lane_result[j])
            );
        end
    endgenerate

    wire [`DATA_WIDTH-1:0] bf16_result = {bf16_lane_result[3], bf16_lane_result[2],
                                           bf16_lane_result[1], bf16_lane_result[0]};

    // EX2 -> EX3 alignment register: delay int16 combinational result by one
    // cycle to match bf16's EX1->EX2->EX3 latency. Also pipe dtype through
    // a second stage so the output mux sees the same instruction's dtype.
    reg [15:0]  int_lane0_r;
    reg [2:0]   dtype_r2;

    always @(posedge clk) begin
        if (rst) begin
            int_lane0_r <= 16'd0;
            dtype_r2    <= 3'd0;
        end else begin
            int_lane0_r <= int_lane0;
            dtype_r2    <= dtype_r1;
        end
    end

    wire [`DATA_WIDTH-1:0] int_result = {4{int_lane0_r}};

    // dtype mux (both paths time-aligned in EX3)
    assign result = (dtype_r2 == `DTYPE_BF16) ? bf16_result : int_result;

endmodule
