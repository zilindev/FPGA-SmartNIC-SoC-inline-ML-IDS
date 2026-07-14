// data_memory.v — 1024 x 64-bit true dual-port BRAM
`timescale 1ns / 1ps
`include "gpu_params.vh"

module data_memory (
    input                            clk,

    // Port A — Core pipeline
    input  [`DMEM_ADDR_W-1:0]       addr,
    input                            wr_en,
    input  [`DATA_WIDTH-1:0]        wr_data,
    output reg [`DATA_WIDTH-1:0]    rd_data,

    // Port B — External host
    input  [`DMEM_ADDR_W-1:0]       addr_b,
    input                            wr_en_b,
    input  [`DATA_WIDTH-1:0]        wr_data_b,
    output reg [`DATA_WIDTH-1:0]    rd_data_b
);

    reg [`DATA_WIDTH-1:0] mem [0:`DMEM_DEPTH-1];

    // Port A: synchronous write + read (read-first)
    always @(posedge clk) begin
        if (wr_en)
            mem[addr] <= wr_data;
        rd_data <= mem[addr];
    end

    // Port B: synchronous write + read (read-first)
    always @(posedge clk) begin
        if (wr_en_b)
            mem[addr_b] <= wr_data_b;
        rd_data_b <= mem[addr_b];
    end

endmodule
