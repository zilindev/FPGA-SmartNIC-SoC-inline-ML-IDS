// writeback_stage.v — Combinational MUX: ALU result or DMEM load
`timescale 1ns / 1ps
`include "gpu_params.vh"

module writeback_stage (
    input                       reg_write_in,
    input                       mem_read,
    input  [`REG_ADDR_W-1:0]    rd_in,
    input  [`DATA_WIDTH-1:0]    alu_result,
    input  [`DATA_WIDTH-1:0]    mem_data,

    output                      reg_write_out,
    output [`REG_ADDR_W-1:0]    rd_out,
    output [`DATA_WIDTH-1:0]    data_out
);

    assign reg_write_out = reg_write_in;
    assign rd_out        = rd_in;
    assign data_out = mem_read ? mem_data : alu_result;

endmodule
