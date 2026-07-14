// simd_alu.v — 4-lane SIMD ALU wrapper (2-cycle pipelined)
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

    wire [15:0] lane_result [0:`SIMD_LANES-1];
    wire [15:0] bf16_lane_result [0:`SIMD_LANES-1];

    genvar i;
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

    // Int16 ALU instances (combinational, EX1)
    generate
        for (i = 0; i < `SIMD_LANES; i = i + 1) begin : lane
            int16_alu u_alu (
                .alu_op (alu_op),
                .a      (effective_a[i*16+15 : i*16]),
                .b      (operand_b[i*16+15 : i*16]),
                .result (lane_result[i])
            );
        end
    endgenerate

    // BF16 ALU instances (2-cycle, result in EX2)
    generate
        for (j = 0; j < `SIMD_LANES; j = j + 1) begin : bf16_lane
            bf16_alu u_bf16 (
                .clk    (clk),
                .rst    (rst),
                .alu_op (alu_op),
                .a      (effective_a[j*16+15 : j*16]),
                .b      (operand_b[j*16+15 : j*16]),
                .result (bf16_lane_result[j])
            );
        end
    endgenerate

    wire [`DATA_WIDTH-1:0] int_result = {lane_result[3], lane_result[2],
                                          lane_result[1], lane_result[0]};

    wire [`DATA_WIDTH-1:0] bf16_result = {bf16_lane_result[3], bf16_lane_result[2],
                                           bf16_lane_result[1], bf16_lane_result[0]};

    // Register int16 result and dtype to align with bf16 2-cycle latency
    reg [`DATA_WIDTH-1:0] int_result_r;
    reg [2:0] dtype_r;

    always @(posedge clk) begin
        if (rst) begin
            int_result_r <= {`DATA_WIDTH{1'b0}};
            dtype_r      <= 3'd0;
        end else begin
            int_result_r <= int_result;
            dtype_r      <= dtype;
        end
    end

    // dtype mux (both paths time-aligned in EX2)
    assign result = (dtype_r == `DTYPE_BF16) ? bf16_result : int_result_r;

endmodule
