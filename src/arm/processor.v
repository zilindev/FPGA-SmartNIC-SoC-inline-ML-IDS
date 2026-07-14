`timescale 1ns / 1ps

// processor.v -- Quad-thread 5-stage ARM pipeline

module processor(
    input clk,
    input rst,

    input         start_stop,       // 1=running, 0=paused

    // IMEM Port B (PCI)
    input         intf_wen_imem,
    input  [8:0]  intf_addr_imem,
    input  [31:0] intf_data_imem,
    output [31:0] intf_dout_imem,

    // DMEM Port B (PCI)
    input         web,
    input  [63:0] dinb,
    input  [7:0]  addrb,
    output [63:0] doutb,

    // External memory port (FIFO BRAM)
    output [7:0]  ext_mem_addr,
    output [63:0] ext_mem_din,
    output        ext_mem_we,
    input  [63:0] ext_mem_dout,

    // Logic Analyzer read port
    input  [10:0] la_rd_addr,
    output [63:0] la_rd_data,

    output [3:0]  dbg_cpsr_0,
    output [3:0]  dbg_cpsr_1,
    output [3:0]  dbg_cpsr_2,
    output [3:0]  dbg_cpsr_3,
    output [1:0]  dbg_thread_id,
    output        all_halted
);

// --- Global Thread Counter ---

reg [1:0] thread_id;

always @(posedge clk) begin
    if (rst)
        thread_id <= 2'd0;
    else if (start_stop)
        thread_id <= thread_id + 2'd1;
end

assign dbg_thread_id = thread_id;

reg [3:0] halted;
assign all_halted = &halted;
wire halt_detected;

// --- IF Stage ---

wire [31:0] if_inst;
wire [10:0] if_pc_plus4;
wire [1:0]  if_thread_id_out;
wire [4:0]  IF_pc_idx;
wire [63:0] branch_addr;
wire        ctrl_branch;
wire [1:0]  id_thread_id;

IF_stage u_if(
    .clk(clk),
    .rst(rst),
    .intf_wen_imem(intf_wen_imem),
    .intf_addr_imem(intf_addr_imem),
    .intf_data_imem(intf_data_imem),
    .intf_dout_imem(intf_dout_imem),
    .data_out(if_inst),
    .pc_plus4_out(if_pc_plus4),
    .thread_id_out(if_thread_id_out),
    .pc_idx_out(IF_pc_idx),
    .thread_id(thread_id),
    .ctrl_branch(ctrl_branch),
    .branch_addr(branch_addr),
    .branch_thread_id(id_thread_id),
    .halted(halted)
);

// --- IF/ID Pipeline Register ---

wire flush = 1'b0; // FGMT eliminates branch flush

wire [31:0] ifid_inst;
wire [10:0] ifid_pc;
wire [1:0]  ifid_thread_id;
wire [4:0]  ID_pc_idx;

ifid_reg u_ifid(
    .clk(clk),
    .rst(rst),
    .flush(flush),
    .i_inst(if_inst),
    .o_inst(ifid_inst),
    .i_pc(if_pc_plus4),
    .o_pc(ifid_pc),
    .i_thread_id(if_thread_id_out),
    .o_thread_id(ifid_thread_id),
    .i_pc_idx(IF_pc_idx),
    .o_pc_idx(ID_pc_idx)
);

// --- Per-thread CPSR Flags ---

reg [3:0] cpsr_0, cpsr_1, cpsr_2, cpsr_3; // {N, Z, C, V}

wire        mem_set_flags;
wire        mem_N, mem_Z, mem_C, mem_V;
wire [1:0]  mem_thread_id;

always @(posedge clk) begin
    if (rst) begin
        cpsr_0 <= 4'd0;
        cpsr_1 <= 4'd0;
        cpsr_2 <= 4'd0;
        cpsr_3 <= 4'd0;
    end else if (mem_set_flags) begin
        case (mem_thread_id)
            2'd0: cpsr_0 <= {mem_N, mem_Z, mem_C, mem_V};
            2'd1: cpsr_1 <= {mem_N, mem_Z, mem_C, mem_V};
            2'd2: cpsr_2 <= {mem_N, mem_Z, mem_C, mem_V};
            2'd3: cpsr_3 <= {mem_N, mem_Z, mem_C, mem_V};
        endcase
    end
end

assign dbg_cpsr_0 = cpsr_0;
assign dbg_cpsr_1 = cpsr_1;
assign dbg_cpsr_2 = cpsr_2;
assign dbg_cpsr_3 = cpsr_3;

reg [3:0] cpsr_selected;
always @(*) begin
    case (ifid_thread_id)
        2'd0: cpsr_selected = cpsr_0;
        2'd1: cpsr_selected = cpsr_1;
        2'd2: cpsr_selected = cpsr_2;
        2'd3: cpsr_selected = cpsr_3;
        default: cpsr_selected = 4'd0;
    endcase
end

wire cpsr_N_id = cpsr_selected[3];
wire cpsr_Z_id = cpsr_selected[2];
wire cpsr_C_id = cpsr_selected[1];
wire cpsr_V_id = cpsr_selected[0];

// --- ID Stage ---

wire [63:0] id_R1, id_R2;
wire [63:0] id_sign_ext;
wire [3:0]  id_WReg1;
wire        id_WRegEn, id_WMemEn, id_RMemEn;
wire        id_ctrl_mem2reg;
wire        id_alusrc;
wire [3:0]  id_ALU_op;
wire        id_set_flags;
wire [4:0]  id_shift_amt;
wire [1:0]  id_shift_type;

wire [63:0] wb_data;
wire [3:0]  wb_waddr;
wire        wb_wen;
wire [1:0]  wb_thread_id;

assign id_thread_id = ifid_thread_id;

ID u_id(
    .clk(clk),
    .rst(rst),
    .i_inst(ifid_inst),
    .waddr(wb_waddr),
    .wdata(wb_data),
    .i_WRegEn(wb_wen),
    .wb_thread_id(wb_thread_id),
    .ifid_pc(ifid_pc),
    .thread_id(ifid_thread_id),
    .r0data(id_R1),
    .r1data(id_R2),
    .o_sign_extended(id_sign_ext),
    .o_WReg1(id_WReg1),
    .o_WRegEn(id_WRegEn),
    .o_WMemEn(id_WMemEn),
    .o_RMemEn(id_RMemEn),
    .ctrl_branch(ctrl_branch),
    .ctrl_mem2reg(id_ctrl_mem2reg),
    .branch_addr(branch_addr),
    .alusrc(id_alusrc),
    .ALU_op(id_ALU_op),
    .o_set_flags(id_set_flags),
    .is_halt(halt_detected),
    .o_shift_amt(id_shift_amt),
    .o_shift_type(id_shift_type),
    .cpsr_N(cpsr_N_id),
    .cpsr_Z(cpsr_Z_id),
    .cpsr_C(cpsr_C_id),
    .cpsr_V(cpsr_V_id)
);

always @(posedge clk) begin
    if (rst)
        halted <= 4'b0000;
    else if (halt_detected)
        halted[id_thread_id] <= 1'b1;
end

// --- ID/EX Pipeline Register ---

wire [63:0] ex_R1, ex_R2, ex_sign_ext;
wire [3:0]  ex_WReg1;
wire        ex_WRegEn, ex_WMemEn, ex_RMemEn;
wire        ex_ctrl_mem2reg;
wire        ex_alusrc;
wire [3:0]  ex_ALU_op;
wire [4:0]  ex_shift_amt;
wire [1:0]  ex_shift_type;
wire        ex_set_flags;
wire [1:0]  ex_thread_id;
wire [4:0]  EX_pc_idx;

idex_reg u_idex(
    .clk(clk),
    .rst(rst),
    .i_R1(id_R1),
    .i_R2(id_R2),
    .i_sign_extended(id_sign_ext),
    .o_R1(ex_R1),
    .o_R2(ex_R2),
    .o_sign_extended(ex_sign_ext),
    .i_WReg1(id_WReg1),
    .i_WRegEn(id_WRegEn),
    .i_WMemEn(id_WMemEn),
    .i_RMemEn(id_RMemEn),
    .i_ctrl_mem2reg(id_ctrl_mem2reg),
    .o_WReg1(ex_WReg1),
    .o_WRegEn(ex_WRegEn),
    .o_WMemEn(ex_WMemEn),
    .o_RMemEn(ex_RMemEn),
    .o_ctrl_mem2reg(ex_ctrl_mem2reg),
    .i_alusrc(id_alusrc),
    .i_ALU_op(id_ALU_op),
    .i_shift_amt(id_shift_amt),
    .i_shift_type(id_shift_type),
    .i_set_flags(id_set_flags),
    .o_alusrc(ex_alusrc),
    .o_ALU_op(ex_ALU_op),
    .o_shift_amt(ex_shift_amt),
    .o_shift_type(ex_shift_type),
    .o_set_flags(ex_set_flags),
    .i_thread_id(ifid_thread_id),
    .o_thread_id(ex_thread_id),
    .i_pc_idx(ID_pc_idx),
    .o_pc_idx(EX_pc_idx)
);

// --- EX Stage ---

wire [63:0] ex_alu_result;
wire [63:0] ex_write_data;
wire        ex_flag_N, ex_flag_Z, ex_flag_C, ex_flag_V;

EX_stage u_ex(
    .R1(ex_R1),
    .R2(ex_R2),
    .sign_extended(ex_sign_ext),
    .alu_src(ex_alusrc),
    .ALU_op(ex_ALU_op),
    .shift_amt(ex_shift_amt),
    .shift_type(ex_shift_type),
    .alu_result(ex_alu_result),
    .write_data(ex_write_data),
    .flag_N(ex_flag_N),
    .flag_Z(ex_flag_Z),
    .flag_C(ex_flag_C),
    .flag_V(ex_flag_V)
);

// --- EX/MEM Pipeline Register ---

wire [63:0] mem_alu_result;
wire [63:0] mem_write_data;
wire [3:0]  mem_WReg1;
wire        mem_WRegEn, mem_WMemEn, mem_RMemEn;
wire        mem_ctrl_mem2reg;
wire [4:0]  MEM_pc_idx;

exmem_reg u_exmem(
    .clk(clk),
    .rst(rst),
    .i_alu_result(ex_alu_result),
    .i_write_data(ex_write_data),
    .o_alu_result(mem_alu_result),
    .o_write_data(mem_write_data),
    .i_set_flags(ex_set_flags),
    .i_N(ex_flag_N),
    .i_Z(ex_flag_Z),
    .i_C(ex_flag_C),
    .i_V(ex_flag_V),
    .o_set_flags(mem_set_flags),
    .o_N(mem_N),
    .o_Z(mem_Z),
    .o_C(mem_C),
    .o_V(mem_V),
    .i_WReg1(ex_WReg1),
    .i_WRegEn(ex_WRegEn),
    .i_WMemEn(ex_WMemEn),
    .i_RMemEn(ex_RMemEn),
    .i_ctrl_mem2reg(ex_ctrl_mem2reg),
    .o_WReg1(mem_WReg1),
    .o_WRegEn(mem_WRegEn),
    .o_WMemEn(mem_WMemEn),
    .o_RMemEn(mem_RMemEn),
    .o_ctrl_mem2reg(mem_ctrl_mem2reg),
    .i_thread_id(ex_thread_id),
    .o_thread_id(mem_thread_id),
    .i_pc_idx(EX_pc_idx),
    .o_pc_idx(MEM_pc_idx)
);

// --- MEM Stage ---

wire [63:0] mem_data_out;
wire [63:0] mem_skip;
wire [3:0]  wb_reg;
wire        wb_reg_en;
wire        wb_ctrl_mem2reg;
wire [1:0]  memwb_thread_id;
wire [4:0]  WB_pc_idx;

data_mem u_mem(
    .clk(clk),
    .reset(rst),
    .addr(mem_alu_result),
    .dina(mem_write_data),
    .WMemEn(mem_WMemEn),
    .douta(mem_data_out),
    .skip_mem(mem_skip),
    .i_WRegEn(mem_WRegEn),
    .i_WReg1(mem_WReg1),
    .i_ctrl_mem2reg(mem_ctrl_mem2reg),
    .o_MRegEn(wb_reg_en),
    .o_WReg1(wb_reg),
    .o_ctrl_mem2reg(wb_ctrl_mem2reg),
    .i_thread_id(mem_thread_id),
    .o_thread_id(memwb_thread_id),
    .i_pc_idx(MEM_pc_idx),
    .o_pc_idx(WB_pc_idx),
    .web(web),
    .dinb(dinb),
    .addrb(addrb),
    .doutb(doutb),
    .ext_mem_addr(ext_mem_addr),
    .ext_mem_din(ext_mem_din),
    .ext_mem_we(ext_mem_we),
    .ext_mem_dout(ext_mem_dout)
);

// --- WB Stage ---

WB u_wb(
    .i_mem_data(mem_data_out),
    .i_skip_mem(mem_skip),
    .dout(wb_data),
    .i_WRegEn(wb_reg_en),
    .i_WReg1(wb_reg),
    .i_ctrl_mem2reg(wb_ctrl_mem2reg),
    .o_WRegEn(wb_wen),
    .o_WReg1(wb_waddr),
    .o_ctrl_mem2reg(),
    .i_thread_id(memwb_thread_id),
    .o_thread_id(wb_thread_id)
);

// --- Logic Analyzer (2048 x 64-bit trace buffer) ---

localparam LA_AW = 11;

reg [LA_AW-1:0] la_wr_ptr;
wire la_event = wb_wen | mem_WMemEn | mem_RMemEn;
wire la_we    = start_stop & la_event;

always @(posedge clk) begin
    if (rst)
        la_wr_ptr <= {LA_AW{1'b0}};
    else if (la_we)
        la_wr_ptr <= la_wr_ptr + 1;
end

// Trace format: {rsv[63:61], mem_tid[60:59], mem_pc[58:54], wb_tid[53:52],
//   wb_pc[51:47], WMemEn[46], RMemEn[45], wb_wen[44], wb_waddr[43:40],
//   mem_addr[39:32], mem_wdata[31:16], wb_data[15:0]}
wire [63:0] la_din = {
    3'b0,
    mem_thread_id,
    MEM_pc_idx,
    wb_thread_id,
    WB_pc_idx,
    mem_WMemEn,
    mem_RMemEn,
    wb_wen,
    wb_waddr,
    mem_alu_result[7:0],
    mem_write_data[15:0],
    wb_data[15:0]
};

logic_analyzer u_la(
    .clk(clk),
    .addra(la_wr_ptr),
    .dina(la_din),
    .wea(la_we),
    .addrb(la_rd_addr),
    .doutb(la_rd_data)
);

endmodule
