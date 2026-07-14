// instruction_memory.v — 1024 x 32-bit true dual-port BRAM
`timescale 1ns / 1ps
`include "gpu_params.vh"

module instruction_memory (
    input                            clk,

    // Port A — Core pipeline (read-only with enable)
    input                            rd_en,
    input  [`IMEM_ADDR_W-1:0]       addr,
    output reg [`INSTR_WIDTH-1:0]   rd_data,

    // Port B — External host (read + write)
    input  [`IMEM_ADDR_W-1:0]       addr_b,
    input                            wr_en_b,
    input  [`INSTR_WIDTH-1:0]       wr_data_b,
    output reg [`INSTR_WIDTH-1:0]   rd_data_b
);

    reg [`INSTR_WIDTH-1:0] mem [0:`IMEM_DEPTH-1];

    // Port A: synchronous read with enable (holds output on stall)
    always @(posedge clk) begin
        if (rd_en)
            rd_data <= mem[addr];
    end

    // Port B: synchronous write + read (read-first)
    always @(posedge clk) begin
        if (wr_en_b)
            mem[addr_b] <= wr_data_b;
        rd_data_b <= mem[addr_b];
    end

endmodule
