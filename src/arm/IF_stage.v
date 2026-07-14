`timescale 1ns / 1ps

// IF_stage.v -- Instruction Fetch, quad-thread FGMT

module IF_stage(
    input         clk,
    input         rst,

    // Port B (PCI)
    input         intf_wen_imem,
    input  [8:0]  intf_addr_imem,
    input  [31:0] intf_data_imem,
    output [31:0] intf_dout_imem,

    output [31:0] data_out,
    output reg [10:0] pc_plus4_out,
    output reg [1:0]  thread_id_out,
    output reg [4:0]  pc_idx_out,

    input  [1:0]  thread_id,

    // Branch control (from ID)
    input         ctrl_branch,
    input  [63:0] branch_addr,
    input  [1:0]  branch_thread_id,

    input  [3:0]  halted
);

    // Per-thread PCs: {thread_id[1:0], offset[8:0]}
    reg [10:0] pc [0:3];

    wire [10:0] current_pc   = pc[thread_id];
    wire [8:0]  current_off  = current_pc[8:0];
    wire [8:0]  next_off     = current_off + 9'd4; // wraps inside region
    wire [10:0] next_pc_seq  = {thread_id, next_off};

    // Branch target confined to branch thread's region
    wire [10:0] next_pc_br   = {branch_thread_id, branch_addr[8:0]};

    always @(posedge clk) begin
        if (rst) begin
            pc[0] <= 11'h000;
            pc[1] <= 11'h200;
            pc[2] <= 11'h400;
            pc[3] <= 11'h600;
            pc_plus4_out  <= 11'd0;
            thread_id_out <= 2'd0;
            pc_idx_out    <= 5'd0;
        end else begin
            if (!halted[thread_id])
                pc[thread_id] <= next_pc_seq;

            if (ctrl_branch)
                pc[branch_thread_id] <= next_pc_br;

            pc_plus4_out  <= next_pc_seq;
            thread_id_out <= thread_id;
            pc_idx_out    <= current_pc[6:2]; // instruction index
        end
    end

    // Instruction Memory (dual-port BRAM)
    inst_mem m_imem (
        .clk   (clk),
        .addra (current_pc[10:2]), // word index
        .dina  (32'd0),
        .wea   (1'b0),
        .douta (data_out),
        .addrb (intf_addr_imem),
        .dinb  (intf_data_imem),
        .web   (intf_wen_imem),
        .doutb (intf_dout_imem)
    );

endmodule
