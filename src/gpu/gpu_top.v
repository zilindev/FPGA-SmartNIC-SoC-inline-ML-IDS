// gpu_top.v
`timescale 1ns / 1ps
`include "gpu_params.vh"

module gpu_top (
    input         clk,
    input         rst,

    input  [31:0] thread_id,
    input  [31:0] block_id,
    input  [31:0] block_dim,

    output        kernel_done,

    // External IMEM interface (Port B)
    input  [`IMEM_ADDR_W-1:0]   ext_imem_addr,
    input                        ext_imem_wr_en,
    input  [`INSTR_WIDTH-1:0]   ext_imem_wr_data,
    output [`INSTR_WIDTH-1:0]   ext_imem_rd_data,

    // External DMEM interface (Port B)
    input  [`DMEM_ADDR_W-1:0]   ext_dmem_addr,
    input                        ext_dmem_wr_en,
    input  [`DATA_WIDTH-1:0]    ext_dmem_wr_data,
    output [`DATA_WIDTH-1:0]    ext_dmem_rd_data
);

    wire        core_halted;

    // Core <-> IMEM
    wire [`IMEM_ADDR_W-1:0] imem_addr;
    wire                    imem_rd_en;
    wire [`INSTR_WIDTH-1:0] imem_data;

    // Core <-> DMEM
    wire [`DMEM_ADDR_W-1:0] dmem_addr;
    wire [`DATA_WIDTH-1:0]  dmem_wr_data;
    wire                    dmem_wr_en;
    wire [`DATA_WIDTH-1:0]  dmem_rd_data;

    assign kernel_done = core_halted;

    // Processor Core (6-stage: IF -> ID -> EX1 -> EX2 -> EX3 -> WB)
    core_top u_core (
        .clk          (clk),
        .rst          (rst),
        .thread_id    (thread_id),
        .block_id     (block_id),
        .block_dim    (block_dim),
        .halted       (core_halted),
        .imem_addr    (imem_addr),
        .imem_rd_en   (imem_rd_en),
        .imem_data    (imem_data),
        .dmem_addr    (dmem_addr),
        .dmem_wr_data (dmem_wr_data),
        .dmem_wr_en   (dmem_wr_en),
        .dmem_rd_data (dmem_rd_data)
    );

    // Instruction Memory (1024 x 32-bit BRAM)
    instruction_memory u_imem (
        .clk       (clk),
        .rd_en     (imem_rd_en),
        .addr      (imem_addr),
        .rd_data   (imem_data),
        .addr_b    (ext_imem_addr),
        .wr_en_b   (ext_imem_wr_en),
        .wr_data_b (ext_imem_wr_data),
        .rd_data_b (ext_imem_rd_data)
    );

    // Data Memory (1024 x 64-bit BRAM)
    data_memory u_dmem (
        .clk       (clk),
        .addr      (dmem_addr),
        .wr_en     (dmem_wr_en),
        .wr_data   (dmem_wr_data),
        .rd_data   (dmem_rd_data),
        .addr_b    (ext_dmem_addr),
        .wr_en_b   (ext_dmem_wr_en),
        .wr_data_b (ext_dmem_wr_data),
        .rd_data_b (ext_dmem_rd_data)
    );

endmodule
