`timescale 1ns / 1ps

// exmem_reg.v -- EX/MEM pipeline register, quad-thread

module exmem_reg(
    input         clk,
    input         rst,

    input  [63:0] i_alu_result,
    input  [63:0] i_write_data,
    output reg [63:0] o_alu_result,
    output reg [63:0] o_write_data,

    // CPSR flags
    input         i_set_flags,
    input         i_N,
    input         i_Z,
    input         i_C,
    input         i_V,
    output reg    o_set_flags,
    output reg    o_N,
    output reg    o_Z,
    output reg    o_C,
    output reg    o_V,

    input  [3:0]  i_WReg1,
    input         i_WRegEn,
    input         i_WMemEn,
    input         i_RMemEn,
    input         i_ctrl_mem2reg,
    output reg [3:0]  o_WReg1,
    output reg        o_WRegEn,
    output reg        o_WMemEn,
    output reg        o_RMemEn,
    output reg        o_ctrl_mem2reg,

    input  [1:0]  i_thread_id,
    output reg [1:0] o_thread_id,

    input  [4:0]  i_pc_idx,
    output reg [4:0] o_pc_idx
);

always @(posedge clk) begin
    if (rst) begin
        o_alu_result   <= 64'd0;
        o_write_data   <= 64'd0;
        o_set_flags    <= 1'b0;
        o_N            <= 1'b0;
        o_Z            <= 1'b0;
        o_C            <= 1'b0;
        o_V            <= 1'b0;
        o_WReg1        <= 4'd0;
        o_WRegEn       <= 1'b0;
        o_WMemEn       <= 1'b0;
        o_RMemEn       <= 1'b0;
        o_ctrl_mem2reg <= 1'b0;
        o_thread_id    <= 2'd0;
        o_pc_idx       <= 5'd0;
    end else begin
        o_alu_result   <= i_alu_result;
        o_write_data   <= i_write_data;
        o_set_flags    <= i_set_flags;
        o_N            <= i_N;
        o_Z            <= i_Z;
        o_C            <= i_C;
        o_V            <= i_V;
        o_WReg1        <= i_WReg1;
        o_WRegEn       <= i_WRegEn;
        o_WMemEn       <= i_WMemEn;
        o_RMemEn       <= i_RMemEn;
        o_ctrl_mem2reg <= i_ctrl_mem2reg;
        o_thread_id    <= i_thread_id;
        o_pc_idx       <= i_pc_idx;
    end
end

endmodule
