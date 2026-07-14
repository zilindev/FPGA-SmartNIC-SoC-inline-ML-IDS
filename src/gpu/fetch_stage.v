// fetch_stage.v — IF/ID pipeline metadata (PC + valid)
`timescale 1ns / 1ps
`include "gpu_params.vh"

module fetch_stage (
    input                        clk,
    input                        rst,
    input                        stall,
    input                        flush,

    input  [`PC_WIDTH-1:0]       pc_in,

    output reg [`PC_WIDTH-1:0]    pc_out,
    output reg                    valid_out
);

    always @(posedge clk) begin
        if (rst || flush) begin
            pc_out    <= {`PC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (!stall) begin
            pc_out    <= pc_in;
            valid_out <= 1'b1;
        end
    end

endmodule
