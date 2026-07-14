`timescale 1ns / 1ps

// fifo_bram — 256x72 True Dual-Port BRAM

module fifo_bram (
    input         clk,
    // Port A
    input  [7:0]  addr_a,
    input  [71:0] din_a,
    input         we_a,
    output reg [71:0] dout_a,
    // Port B
    input  [7:0]  addr_b,
    input  [71:0] din_b,
    input         we_b,
    output reg [71:0] dout_b
);

    reg [71:0] mem [0:255];

    // Port A
    always @(posedge clk) begin
        if (we_a)
            mem[addr_a] <= din_a;
        dout_a <= mem[addr_a];
    end

    // Port B
    always @(posedge clk) begin
        if (we_b)
            mem[addr_b] <= din_b;
        dout_b <= mem[addr_b];
    end

endmodule
