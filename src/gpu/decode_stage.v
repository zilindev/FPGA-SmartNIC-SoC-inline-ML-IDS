// decode_stage.v — Instruction decoder + register file wrapper
`timescale 1ns / 1ps
`include "gpu_params.vh"

module decode_stage (
    input                       clk,
    input                       rst,
    input                       stall,
    input                       flush,

    // From IF/ID
    input  [`INSTR_WIDTH-1:0]   instruction,
    input  [`PC_WIDTH-1:0]      pc_in,
    input                       valid_in,

    // Thread identity for special registers
    input  [31:0]               thread_id,
    input  [31:0]               block_id,
    input  [31:0]               block_dim,

    // Writeback port
    input                       wb_reg_write,
    input  [`REG_ADDR_W-1:0]    wb_rd,
    input  [`DATA_WIDTH-1:0]    wb_data,

    // Decoded outputs
    output [3:0]                alu_op_out,
    output                      reg_write_out,
    output                      mem_read_out,
    output                      mem_write_out,
    output                      is_fma_out,
    output                      use_imm_out,
    output [2:0]                dtype_out,
    output                      is_halt_out,
    output [1:0]                branch_type_out,

    output [`REG_ADDR_W-1:0]    rd_out,
    output [`REG_ADDR_W-1:0]    rs1_addr_out,
    output [`REG_ADDR_W-1:0]    rs2_addr_out,
    output [`REG_ADDR_W-1:0]    rs3_addr_out,

    output [`DATA_WIDTH-1:0]    rs1_data_out,
    output [`DATA_WIDTH-1:0]    rs2_data_out,
    output [`DATA_WIDTH-1:0]    rs3_data_out,

    output [`DATA_WIDTH-1:0]    imm_ext_out,
    output [`PC_WIDTH-1:0]      pc_out
);

    // Decoder outputs
    wire [3:0]               dec_alu_op;
    wire                     dec_reg_write;
    wire                     dec_mem_read;
    wire                     dec_mem_write;
    wire                     dec_is_fma;
    wire [1:0]               dec_branch_type;
    wire                     dec_use_imm;
    wire [2:0]               dec_dtype;
    wire                     dec_is_halt;
    wire [`REG_ADDR_W-1:0]   dec_rd;
    wire [`REG_ADDR_W-1:0]   dec_rs1_addr;
    wire [`REG_ADDR_W-1:0]   dec_rs2_addr;
    wire [`REG_ADDR_W-1:0]   dec_rs3_addr;
    wire [`DATA_WIDTH-1:0]   dec_imm_ext;

    instruction_decoder u_decoder (
        .instruction (instruction),
        .valid       (valid_in),
        .alu_op      (dec_alu_op),
        .reg_write   (dec_reg_write),
        .mem_read    (dec_mem_read),
        .mem_write   (dec_mem_write),
        .is_fma      (dec_is_fma),
        .branch_type (dec_branch_type),
        .use_imm     (dec_use_imm),
        .dtype       (dec_dtype),
        .is_halt     (dec_is_halt),
        .rd          (dec_rd),
        .rs1_addr    (dec_rs1_addr),
        .rs2_addr    (dec_rs2_addr),
        .rs3_addr    (dec_rs3_addr),
        .imm_ext     (dec_imm_ext)
    );

    // Branch-aware read address muxing
    wire is_branch = (dec_branch_type != `BR_NONE);

    // Port 1: branches read ISA rs1 from rd field
    wire [`REG_ADDR_W-1:0] rf_rd_addr1 = is_branch ? dec_rd : dec_rs1_addr;

    // Port 2: branches read ISA rs2; ST reads store data from rd field
    wire [`REG_ADDR_W-1:0] rf_rd_addr2 = is_branch    ? dec_rs1_addr :
                                          dec_mem_write ? dec_rd :
                                          dec_rs2_addr;

    register_file u_regfile (
        .clk         (clk),
        .rst         (rst),
        .thread_id   (thread_id),
        .block_id    (block_id),
        .block_dim   (block_dim),
        .wr_en       (wb_reg_write),
        .wr_addr     (wb_rd),
        .wr_data     (wb_data),
        .rd_addr1    (rf_rd_addr1),
        .rd_addr2    (rf_rd_addr2),
        .rd_addr3    (dec_rs3_addr),
        .rd_data1    (rs1_data_out),
        .rd_data2    (rs2_data_out),
        .rd_data3    (rs3_data_out)
    );

    // Pass-through from decoder
    assign alu_op_out      = dec_alu_op;
    assign reg_write_out   = dec_reg_write;
    assign mem_read_out    = dec_mem_read;
    assign mem_write_out   = dec_mem_write;
    assign is_fma_out      = dec_is_fma;
    assign use_imm_out     = dec_use_imm;
    assign dtype_out       = dec_dtype;
    assign is_halt_out     = dec_is_halt;
    assign branch_type_out = dec_branch_type;

    assign rd_out          = dec_rd;
    assign rs1_addr_out    = rf_rd_addr1;
    assign rs2_addr_out    = rf_rd_addr2;
    assign rs3_addr_out    = dec_rs3_addr;

    assign imm_ext_out     = dec_imm_ext;
    assign pc_out          = pc_in;

endmodule
