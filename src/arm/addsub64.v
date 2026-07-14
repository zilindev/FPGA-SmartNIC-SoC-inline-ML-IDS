`timescale 1ns / 1ps

// addsub64.v -- 64-bit adder/subtractor with carry-out

module addsub64 (
    input  [63:0] A,
    input  [63:0] B,
    input         sub_mode, // 0=add, 1=subtract
    output [63:0] S,
    output        overflow  // carry-out from MSB
);

    wire [63:0] B_eff = B ^ {64{sub_mode}};
    wire [64:0] sum   = {1'b0, A} + {1'b0, B_eff} + {64'd0, sub_mode};

    assign S        = sum[63:0];
    assign overflow = sum[64];

endmodule
