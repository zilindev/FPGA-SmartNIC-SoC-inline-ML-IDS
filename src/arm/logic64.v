`timescale 1ns / 1ps

// logic64.v -- 64-bit AND, OR, XNOR

module logic64 (
    input  [63:0] A,
    input  [63:0] B,
    output [63:0] AND_out,
    output [63:0] OR_out,
    output [63:0] XNOR_out
);

    assign AND_out  = A & B;
    assign OR_out   = A | B;
    assign XNOR_out = ~(A ^ B);

endmodule
