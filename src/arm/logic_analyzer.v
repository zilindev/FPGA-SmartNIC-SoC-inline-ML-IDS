`timescale 1ns / 1ps

// logic_analyzer.v -- 2048 x 64-bit trace buffer (dual-port BRAM)

module logic_analyzer(
    input         clk,
    // Port A: write (trace capture)
    input  [10:0] addra,
    input  [63:0] dina,
    input         wea,
    // Port B: read (PCI readback)
    input  [10:0] addrb,
    output reg [63:0] doutb
);

    reg [63:0] mem [0:2047];

    always @(posedge clk) begin
        if (wea)
            mem[addra] <= dina;
    end

    always @(posedge clk) begin
        doutb <= mem[addrb];
    end

endmodule
