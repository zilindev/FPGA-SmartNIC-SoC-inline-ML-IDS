`timescale 1ns / 1ps

// inst_mem.v -- 32-bit x 512 dual-port BRAM instruction memory

module inst_mem(
    input         clk,

    // Port A: CPU fetch
    input  [8:0]  addra,
    input  [31:0] dina,
    input         wea,
    output reg [31:0] douta,

    // Port B: PCI load/readback
    input  [8:0]  addrb,
    input  [31:0] dinb,
    input         web,
    output reg [31:0] doutb
);

    reg [31:0] mem [0:511];

    // Port A
    always @(posedge clk) begin
        if (wea)
            mem[addra] <= dina;
        douta <= mem[addra];
    end

    // Port B
    always @(posedge clk) begin
        if (web)
            mem[addrb] <= dinb;
        doutb <= mem[addrb];
    end

endmodule
