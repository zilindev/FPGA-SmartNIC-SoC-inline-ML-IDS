`timescale 1ns / 1ps

// data_mem.v -- MEM stage with dual-port DMEM + external FIFO BRAM access
//
// Address decode (addra = addr[7:0]):
//   0x00-0x7F  internal DMEM (128 x 64-bit, shared across threads)
//   0x80-0xEF  external FIFO BRAM window -> ext_mem_* port
//   0xF0-0xFF  control-register alias window -> reg_alias_* port (Milestone 4)

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

    // External memory (FIFO BRAM), active when addra[7]=1 && addra[7:4]!=4'hF
    output [7:0]  ext_mem_addr,
    output [63:0] ext_mem_din,
    output        ext_mem_we,
    input  [63:0] ext_mem_dout,

    // Register-alias window, active when addra[7:4]==4'hF (Milestone 4)
    output [3:0]  reg_alias_sel,
    output [63:0] reg_alias_wdata,
    output        reg_alias_we,
    input  [63:0] reg_alias_rdata
);

    wire [7:0] addra = addr[7:0];

    // Address decode: 0xF0-0xFF carves out of the 0x80-0xFF FIFO window
    wire addr_is_reg = (addra[7:4] == 4'hF);
    wire addr_is_ext = addra[7] & ~addr_is_reg;

    // External FIFO BRAM port
    assign ext_mem_addr = {1'b0, addra[6:0]}; // CPU 0x80-0xEF -> FIFO 0x00-0x6F
    assign ext_mem_din  = dina;
    assign ext_mem_we   = WMemEn & addr_is_ext;

    // Register-alias port
    assign reg_alias_sel   = addra[3:0];
    assign reg_alias_wdata = dina;
    assign reg_alias_we    = WMemEn & addr_is_reg;

    // Track previous-cycle access type for output mux
    reg was_ext;
    reg was_reg;
    always @(posedge clk) begin
        if (reset) begin
            was_ext <= 1'b0;
            was_reg <= 1'b0;
        end else begin
            was_ext <= addr_is_ext;
            was_reg <= addr_is_reg;
        end
    end

    // reg_alias_rdata is combinational in the wrapper (mux from cpu_reg_sel).
    // Capture it on the same edge that latches was_reg so the next cycle's
    // unrelated cpu_reg_sel does not drive our was_reg-qualified douta.
    reg [63:0] reg_alias_rdata_r;
    always @(posedge clk) begin
        if (reset) reg_alias_rdata_r <= 64'd0;
        else       reg_alias_rdata_r <= reg_alias_rdata;
    end

    // 64-bit x 256 DMEM (shared across threads)
    reg [63:0] mem [0:255];

    reg [63:0] dmem_rd;

    // Port A: CPU writes only land in internal DMEM (addr 0x00-0x7F)
    always @(posedge clk) begin
        if (WMemEn && !addr_is_ext && !addr_is_reg)
            mem[addra] <= dina;
        dmem_rd <= mem[addra];
    end

    // Output mux: alias rdata / external FIFO BRAM / internal DMEM
    always @(*) douta = was_reg ? reg_alias_rdata_r
                      : (was_ext ? ext_mem_dout : dmem_rd);

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
