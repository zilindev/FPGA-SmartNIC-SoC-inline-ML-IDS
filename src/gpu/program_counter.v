// program_counter.v
`timescale 1ns / 1ps
`include "gpu_params.vh"

module program_counter (
    input                    clk,
    input                    rst,
    input                    stall,
    input                    branch_en,
    input  [`PC_WIDTH-1:0]   branch_target,
    output reg [`PC_WIDTH-1:0] pc_out
);

    // Priority: rst > branch > stall > increment
    always @(posedge clk) begin
        if (rst) begin
            pc_out <= {`PC_WIDTH{1'b0}};
        end else if (branch_en) begin
            pc_out <= branch_target;
        end else if (stall) begin
            pc_out <= pc_out;
        end else begin
            pc_out <= pc_out + 1'b1;
        end
    end

endmodule
