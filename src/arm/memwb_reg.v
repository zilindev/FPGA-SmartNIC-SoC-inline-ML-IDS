`timescale 1ns / 1ps

// memwb_reg.v -- WB stage (combinational mux)

module WB(
    input  [63:0] i_mem_data,
    input  [63:0] i_skip_mem, // ALU result
    output [63:0] dout,

    input         i_WRegEn,
    input  [3:0]  i_WReg1,
    input         i_ctrl_mem2reg,
    output        o_ctrl_mem2reg,
    output        o_WRegEn,
    output [3:0]  o_WReg1,

    input  [1:0]  i_thread_id,
    output [1:0]  o_thread_id
);

assign o_WRegEn       = i_WRegEn;
assign o_WReg1        = i_WReg1;
assign o_ctrl_mem2reg = i_ctrl_mem2reg;
assign o_thread_id    = i_thread_id;

assign dout = (o_ctrl_mem2reg) ? i_mem_data : i_skip_mem;

endmodule
