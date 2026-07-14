`timescale 1ns / 1ps

// ifid_reg.v -- IF/ID pipeline register, quad-thread

module ifid_reg(
    input clk,
    input rst,
    input flush,

    input  [31:0] i_inst,
    output reg [31:0] o_inst,

    input  [10:0] i_pc,
    output reg [10:0] o_pc,

    input  [1:0] i_thread_id,
    output reg [1:0] o_thread_id,

    input  [4:0] i_pc_idx,
    output reg [4:0] o_pc_idx
);

always @(posedge clk) begin
    if (rst) begin
        o_inst      <= 32'b0;
        o_pc        <= 11'd0;
        o_thread_id <= 2'd0;
        o_pc_idx    <= 5'd0;
    end
    else if (flush) begin
        o_inst      <= 32'b0;
        o_pc        <= 11'd0;
        o_thread_id <= 2'd0;
        o_pc_idx    <= 5'd0;
    end
    else begin
        o_inst      <= i_inst;
        o_pc        <= i_pc;
        o_thread_id <= i_thread_id;
        o_pc_idx    <= i_pc_idx;
    end
end

endmodule
