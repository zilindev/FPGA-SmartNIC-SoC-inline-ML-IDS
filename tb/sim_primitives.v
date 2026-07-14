// sim_primitives.v — Xilinx gate primitives for iverilog simulation
// These are built into ISE but need explicit definitions for iverilog.

`timescale 1ns / 1ps

module XOR2(input I0, input I1, output O);
    assign O = I0 ^ I1;
endmodule

module AND2(input I0, input I1, output O);
    assign O = I0 & I1;
endmodule

module OR3(input I0, input I1, input I2, output O);
    assign O = I0 | I1 | I2;
endmodule
