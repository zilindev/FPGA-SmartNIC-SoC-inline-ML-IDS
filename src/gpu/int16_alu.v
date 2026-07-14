// int16_alu.v — Single-lane combinational 16-bit integer ALU
`timescale 1ns / 1ps
`include "gpu_params.vh"

module int16_alu (
    input  [3:0]          alu_op,
    input  [15:0]         a,
    input  [15:0]         b,
    output reg [15:0]     result
);

    wire [31:0] product = a * b; // infers mult18x18

    always @(*) begin
        case (alu_op)
            `ALU_ADD:  result = a + b;
            `ALU_SUB:  result = a - b;
            `ALU_MUL:  result = product[15:0];
            `ALU_MAX:  result = ($signed(a) > $signed(b)) ? a : b;
            `ALU_MOV:  result = a;
            `ALU_BCAST: result = a;
            `ALU_RELU: result = a[15] ? 16'h0000 : a;
            default:   result = 16'h0000;
        endcase
    end

endmodule
