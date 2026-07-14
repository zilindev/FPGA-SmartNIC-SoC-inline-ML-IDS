// pipeline_reg_ex.v — EX-stage control pipeline register (reusable)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module pipeline_reg_ex (
    input                       clk,
    input                       rst,
    input                       flush,

    input                       reg_write_in,
    input                       mem_read_in,
    input  [`REG_ADDR_W-1:0]    rd_in,
    input                       is_halt_in,

    output reg                  reg_write_out,
    output reg                  mem_read_out,
    output reg [`REG_ADDR_W-1:0]  rd_out,
    output reg                  is_halt_out
);

    always @(posedge clk) begin
        if (rst || flush) begin
            reg_write_out <= 1'b0;
            mem_read_out  <= 1'b0;
            rd_out        <= {`REG_ADDR_W{1'b0}};
            is_halt_out   <= 1'b0;
        end else begin
            reg_write_out <= reg_write_in;
            mem_read_out  <= mem_read_in;
            rd_out        <= rd_in;
            is_halt_out   <= is_halt_in;
        end
    end

endmodule
