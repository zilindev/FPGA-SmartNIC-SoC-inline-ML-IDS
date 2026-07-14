// pipeline_reg_idex.v — ID/EX pipeline register
`timescale 1ns / 1ps
`include "gpu_params.vh"

module pipeline_reg_idex (
    input                       clk,
    input                       rst,
    input                       stall,
    input                       flush,

    input  [3:0]                alu_op_in,
    input                       reg_write_in,
    input                       mem_read_in,
    input                       mem_write_in,
    input                       is_fma_in,
    input                       use_imm_in,
    input  [2:0]                dtype_in,
    input                       is_halt_in,
    input  [1:0]                branch_type_in,
    input  [`REG_ADDR_W-1:0]    rd_in,
    input  [`DATA_WIDTH-1:0]    rs1_data_in,
    input  [`DATA_WIDTH-1:0]    rs2_data_in,
    input  [`DATA_WIDTH-1:0]    rs3_data_in,
    input  [`DATA_WIDTH-1:0]    imm_ext_in,
    input  [`PC_WIDTH-1:0]      pc_in,

    output reg [3:0]            alu_op_out,
    output reg                  reg_write_out,
    output reg                  mem_read_out,
    output reg                  mem_write_out,
    output reg                  is_fma_out,
    output reg                  use_imm_out,
    output reg [2:0]            dtype_out,
    output reg                  is_halt_out,
    output reg [1:0]            branch_type_out,
    output reg [`REG_ADDR_W-1:0]  rd_out,
    output reg [`DATA_WIDTH-1:0]  rs1_data_out,
    output reg [`DATA_WIDTH-1:0]  rs2_data_out,
    output reg [`DATA_WIDTH-1:0]  rs3_data_out,
    output reg [`DATA_WIDTH-1:0]  imm_ext_out,
    output reg [`PC_WIDTH-1:0]    pc_out
);

    always @(posedge clk) begin
        if (rst || flush) begin
            alu_op_out      <= 4'd0;
            reg_write_out   <= 1'b0;
            mem_read_out    <= 1'b0;
            mem_write_out   <= 1'b0;
            is_fma_out      <= 1'b0;
            use_imm_out     <= 1'b0;
            dtype_out       <= 3'd0;
            is_halt_out     <= 1'b0;
            branch_type_out <= `BR_NONE;
            rd_out          <= {`REG_ADDR_W{1'b0}};
            rs1_data_out    <= {`DATA_WIDTH{1'b0}};
            rs2_data_out    <= {`DATA_WIDTH{1'b0}};
            rs3_data_out    <= {`DATA_WIDTH{1'b0}};
            imm_ext_out     <= {`DATA_WIDTH{1'b0}};
            pc_out          <= {`PC_WIDTH{1'b0}};
        end else if (!stall) begin
            alu_op_out      <= alu_op_in;
            reg_write_out   <= reg_write_in;
            mem_read_out    <= mem_read_in;
            mem_write_out   <= mem_write_in;
            is_fma_out      <= is_fma_in;
            use_imm_out     <= use_imm_in;
            dtype_out       <= dtype_in;
            is_halt_out     <= is_halt_in;
            branch_type_out <= branch_type_in;
            rd_out          <= rd_in;
            rs1_data_out    <= rs1_data_in;
            rs2_data_out    <= rs2_data_in;
            rs3_data_out    <= rs3_data_in;
            imm_ext_out     <= imm_ext_in;
            pc_out          <= pc_in;
        end
    end

endmodule
