`timescale 1ns / 1ps

// idex_reg.v -- ID/EX pipeline register, quad-thread

module idex_reg(
    input clk,
    input rst,

    input      [63:0] i_R1,
    input      [63:0] i_R2,
    input      [63:0] i_sign_extended,
    output reg [63:0] o_R1,
    output reg [63:0] o_R2,
    output reg [63:0] o_sign_extended,

    input      [3:0]  i_WReg1,
    input              i_WRegEn,
    input              i_WMemEn,
    input              i_RMemEn,
    input              i_ctrl_mem2reg,
    output reg [3:0]  o_WReg1,
    output reg        o_WRegEn,
    output reg        o_WMemEn,
    output reg        o_RMemEn,
    output reg        o_ctrl_mem2reg,

    input              i_alusrc,
    input      [3:0]  i_ALU_op,
    input      [4:0]  i_shift_amt,
    input      [1:0]  i_shift_type,
    input              i_set_flags,

    output reg        o_alusrc,
    output reg [3:0]  o_ALU_op,
    output reg [4:0]  o_shift_amt,
    output reg [1:0]  o_shift_type,
    output reg        o_set_flags,

    input      [1:0]  i_thread_id,
    output reg [1:0]  o_thread_id,

    input      [4:0]  i_pc_idx,
    output reg [4:0]  o_pc_idx
);

always @(posedge clk) begin
    if (rst) begin
        o_R1            <= 64'd0;
        o_R2            <= 64'd0;
        o_sign_extended <= 64'd0;

        o_WReg1         <= 4'd0;
        o_WRegEn        <= 1'b0;
        o_WMemEn        <= 1'b0;
        o_RMemEn        <= 1'b0;
        o_ctrl_mem2reg  <= 1'b0;

        o_alusrc        <= 1'b0;
        o_ALU_op        <= 4'd0;
        o_shift_amt     <= 5'd0;
        o_shift_type    <= 2'd0;
        o_set_flags     <= 1'b0;

        o_thread_id     <= 2'd0;
        o_pc_idx        <= 5'd0;
    end else begin
        o_R1            <= i_R1;
        o_R2            <= i_R2;
        o_sign_extended <= i_sign_extended;

        o_WReg1         <= i_WReg1;
        o_WRegEn        <= i_WRegEn;
        o_WMemEn        <= i_WMemEn;
        o_RMemEn        <= i_RMemEn;
        o_ctrl_mem2reg  <= i_ctrl_mem2reg;

        o_alusrc        <= i_alusrc;
        o_ALU_op        <= i_ALU_op;
        o_shift_amt     <= i_shift_amt;
        o_shift_type    <= i_shift_type;
        o_set_flags     <= i_set_flags;

        o_thread_id     <= i_thread_id;
        o_pc_idx        <= i_pc_idx;
    end
end

endmodule
