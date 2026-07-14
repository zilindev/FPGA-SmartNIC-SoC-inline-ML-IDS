`timescale 1ns / 1ps

// data_mem.v -- MEM stage with dual-port DMEM + external FIFO BRAM access

module data_mem(
    input         clk,
    input         reset,

    // Port A: CPU datapath
    input  [63:0] addr,
    input  [63:0] dina,
    input         WMemEn,
    output reg [63:0] douta,
    output reg [63:0] skip_mem,

    // MEM/WB control pass-through
    input         i_WRegEn,
    input  [3:0]  i_WReg1,
    input         i_ctrl_mem2reg,
    output reg    o_MRegEn,
    output reg [3:0] o_WReg1,
    output reg    o_ctrl_mem2reg,

    input  [1:0]  i_thread_id,
    output reg [1:0] o_thread_id,

    input  [4:0]  i_pc_idx,
    output reg [4:0] o_pc_idx,

    // Port B: PCI
    input         web,
    input  [63:0] dinb,
    input  [7:0]  addrb,
    output reg [63:0] doutb,

    // External memory (FIFO BRAM), active when addr[7]=1
    output [7:0]  ext_mem_addr,
    output [63:0] ext_mem_din,
    output        ext_mem_we,
    input  [63:0] ext_mem_dout
);

    wire [7:0] addra = addr[7:0];

    // addr[7] selects ext FIFO BRAM vs internal DMEM
    wire addr_is_ext = addra[7];

    assign ext_mem_addr = {1'b0, addra[6:0]}; // CPU 128-255 -> FIFO 0-127
    assign ext_mem_din  = dina;
    assign ext_mem_we   = WMemEn & addr_is_ext;

    // Track previous-cycle external access for output mux
    reg was_ext;
    always @(posedge clk) begin
        if (reset) was_ext <= 1'b0;
        else       was_ext <= addr_is_ext;
    end

    // 64-bit x 256 DMEM (shared across threads)
    reg [63:0] mem [0:255];

    reg [63:0] dmem_rd;

    // Port A: CPU (only writes internal when addr[7]=0)
    always @(posedge clk) begin
        if (WMemEn && !addr_is_ext)
            mem[addra] <= dina;
        dmem_rd <= mem[addra];
    end

    // Output mux: internal DMEM vs external FIFO BRAM
    always @(*) douta = was_ext ? ext_mem_dout : dmem_rd;

    // Port B: PCI
    always @(posedge clk) begin
        if (web)
            mem[addrb] <= dinb;
        doutb <= mem[addrb];
    end

    // ALU result pass-through (registered)
    always @(posedge clk) begin
        skip_mem <= addr;
    end

    // MEM/WB pipeline register
    always @(posedge clk) begin
        o_WReg1        <= i_WReg1;
        o_MRegEn       <= i_WRegEn;
        o_ctrl_mem2reg <= i_ctrl_mem2reg;
        o_thread_id    <= i_thread_id;
        o_pc_idx       <= i_pc_idx;
    end

endmodule
