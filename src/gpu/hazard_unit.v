// hazard_unit.v — Stall-only RAW hazard detection (4 stages)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module hazard_unit (
    input  [`REG_ADDR_W-1:0]    de_rs1,
    input  [`REG_ADDR_W-1:0]    de_rs2,
    input  [`REG_ADDR_W-1:0]    de_rs3,
    input                       de_is_fma,

    input  [`REG_ADDR_W-1:0]    ex1_rd,
    input                       ex1_reg_write,

    input  [`REG_ADDR_W-1:0]    ex2_rd,
    input                       ex2_reg_write,

    input  [`REG_ADDR_W-1:0]    ex3_rd,
    input                       ex3_reg_write,

    input  [`REG_ADDR_W-1:0]    wb_rd,
    input                       wb_reg_write,

    output                      stall
);

    wire ex1_hazard = ex1_reg_write && (ex1_rd != `REG_ZERO) &&
                      ((ex1_rd == de_rs1) || (ex1_rd == de_rs2) ||
                       (de_is_fma && (ex1_rd == de_rs3)));

    wire ex2_hazard = ex2_reg_write && (ex2_rd != `REG_ZERO) &&
                      ((ex2_rd == de_rs1) || (ex2_rd == de_rs2) ||
                       (de_is_fma && (ex2_rd == de_rs3)));

    wire ex3_hazard = ex3_reg_write && (ex3_rd != `REG_ZERO) &&
                      ((ex3_rd == de_rs1) || (ex3_rd == de_rs2) ||
                       (de_is_fma && (ex3_rd == de_rs3)));

    wire wb_hazard = wb_reg_write && (wb_rd != `REG_ZERO) &&
                     ((wb_rd == de_rs1) || (wb_rd == de_rs2) ||
                      (de_is_fma && (wb_rd == de_rs3)));

    assign stall = (ex1_hazard || ex2_hazard || ex3_hazard || wb_hazard);

endmodule
