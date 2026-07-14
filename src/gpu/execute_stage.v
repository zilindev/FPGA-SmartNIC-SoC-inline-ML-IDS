// execute_stage.v — SIMD ALU (2-cycle) | FMA (3-cycle) | LD/ST path
`timescale 1ns / 1ps
`include "gpu_params.vh"

module execute_stage (
    input                       clk,
    input                       rst,

    input  [3:0]                alu_op,
    input                       is_fma,
    input                       use_imm,
    input  [2:0]                dtype,
    input                       mem_write,

    input  [`DATA_WIDTH-1:0]    rs1_data,
    input  [`DATA_WIDTH-1:0]    rs2_data,
    input  [`DATA_WIDTH-1:0]    rs3_data,
    input  [`DATA_WIDTH-1:0]    imm_ext,

    output [`DATA_WIDTH-1:0]    result,       // valid in EX3

    // DMEM interface (active in EX3)
    output [`DMEM_ADDR_W-1:0]   dmem_addr,
    output [`DATA_WIDTH-1:0]    dmem_wr_data,
    output                      dmem_wr_en
);

    // EX1: Operand B MUX
    wire [`DATA_WIDTH-1:0] op_b = use_imm ? imm_ext : rs2_data;

    // SIMD ALU (2-cycle pipelined)
    wire [`DATA_WIDTH-1:0] simd_result;

    simd_alu u_simd_alu (
        .clk       (clk),
        .rst       (rst),
        .alu_op    (alu_op),
        .dtype     (dtype),
        .operand_a (rs1_data),
        .operand_b (op_b),
        .result    (simd_result)
    );

    // 4-lane BF16 FMA (3-cycle pipeline)
    wire [15:0] fma_lane_result [0:`SIMD_LANES-1];
    genvar k;
    generate
        for (k = 0; k < `SIMD_LANES; k = k + 1) begin : fma_lane
            bf16_fma_unit u_fma (
                .clk    (clk),
                .rst    (rst),
                .a      (rs1_data[k*16+15 : k*16]),
                .b      (rs2_data[k*16+15 : k*16]),
                .c      (rs3_data[k*16+15 : k*16]),
                .result (fma_lane_result[k])
            );
        end
    endgenerate
    wire [`DATA_WIDTH-1:0] fma_result = {fma_lane_result[3], fma_lane_result[2],
                                          fma_lane_result[1], fma_lane_result[0]};

    // EX1 -> EX2 pipeline register
    reg                      is_fma_r;
    reg                      mem_write_r;
    reg [`DATA_WIDTH-1:0]    rs2_data_r;
    reg [`DMEM_ADDR_W-1:0]   mem_addr_r;

    always @(posedge clk) begin
        if (rst) begin
            is_fma_r    <= 1'b0;
            mem_write_r <= 1'b0;
            rs2_data_r  <= {`DATA_WIDTH{1'b0}};
            mem_addr_r  <= {`DMEM_ADDR_W{1'b0}};
        end else begin
            is_fma_r    <= is_fma;
            mem_write_r <= mem_write;
            rs2_data_r  <= rs2_data;
            mem_addr_r  <= rs1_data[`DMEM_ADDR_W-1:0] + imm_ext[`DMEM_ADDR_W-1:0];
        end
    end

    // EX2 -> EX3 pipeline register (aligns SIMD with 3-cycle FMA)
    reg                      is_fma_r2;
    reg                      mem_write_r2;
    reg [`DATA_WIDTH-1:0]    rs2_data_r2;
    reg [`DMEM_ADDR_W-1:0]   mem_addr_r2;
    reg [`DATA_WIDTH-1:0]    simd_result_r;

    always @(posedge clk) begin
        if (rst) begin
            is_fma_r2    <= 1'b0;
            mem_write_r2 <= 1'b0;
            rs2_data_r2  <= {`DATA_WIDTH{1'b0}};
            mem_addr_r2  <= {`DMEM_ADDR_W{1'b0}};
            simd_result_r <= {`DATA_WIDTH{1'b0}};
        end else begin
            is_fma_r2    <= is_fma_r;
            mem_write_r2 <= mem_write_r;
            rs2_data_r2  <= rs2_data_r;
            mem_addr_r2  <= mem_addr_r;
            simd_result_r <= simd_result;
        end
    end

    // EX3: Result MUX
    assign result = is_fma_r2 ? fma_result : simd_result_r;

    // EX3: DMEM interface
    assign dmem_addr    = mem_addr_r2;
    assign dmem_wr_data = rs2_data_r2;
    assign dmem_wr_en   = mem_write_r2;

endmodule
