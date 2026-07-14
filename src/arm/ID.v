`timescale 1ns / 1ps

// ID.v -- Instruction Decode / Register File, quad-thread

module ID(
    input             clk,
    input             rst,
    input      [31:0] i_inst,

    // Writeback
    input      [3:0]  waddr,
    input      [63:0] wdata,
    input             i_WRegEn,
    input      [1:0]  wb_thread_id,

    input      [10:0] ifid_pc,
    input      [1:0]  thread_id,

    output     [63:0] r0data,
    output     [63:0] r1data,
    output     [63:0] o_sign_extended,

    output     [3:0]  o_WReg1,
    output reg        o_WRegEn,
    output reg        o_WMemEn,
    output reg        o_RMemEn,
    output reg        ctrl_branch,
    output reg        ctrl_mem2reg,
    output     [63:0] branch_addr,
    output reg        alusrc,
    output reg [3:0]  ALU_op,
    output reg        o_set_flags,

    // Barrel shifter control
    output     [4:0]  o_shift_amt,
    output     [1:0]  o_shift_type,

    output            is_halt,

    // Per-thread CPSR flags
    input             cpsr_N,
    input             cpsr_Z,
    input             cpsr_C,
    input             cpsr_V
);

    // Register File: 64x64-bit (4 threads x 16 regs)
    reg [63:0] register_file [0:63];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 64; i = i + 1)
                register_file[i] <= 64'd0;
        end else begin
            if (i_WRegEn)
                register_file[{wb_thread_id, waddr}] <= wdata;
        end
    end

    // --- Instruction Field Extraction ---
    wire [3:0]  cond       = i_inst[31:28];
    wire [1:0]  op_class   = i_inst[27:26];
    wire        I_bit      = i_inst[25];
    wire [3:0]  opcode     = i_inst[24:21];
    wire        U_bit      = i_inst[23];
    wire        S_bit      = i_inst[20];
    wire        L_bit      = i_inst[20];
    wire [3:0]  Rn         = i_inst[19:16];
    wire [3:0]  Rd         = i_inst[15:12];

    wire [3:0]  Rm         = i_inst[3:0];
    wire [4:0]  shift_amt  = i_inst[11:7];
    wire [1:0]  shift_type = i_inst[6:5];

    wire [3:0]  rotate     = i_inst[11:8];
    wire [7:0]  imm8       = i_inst[7:0];
    wire [11:0] offset12   = i_inst[11:0];

    // --- Rotated Immediate ---
    reg  [31:0] imm32;
    wire [4:0]  rot_amt    = {rotate, 1'b0};
    wire [31:0] imm32_raw  = {24'b0, imm8};

    always @(*) begin
        if (rotate == 4'd0)
            imm32 = imm32_raw;
        else
            imm32 = (imm32_raw >> rot_amt) | (imm32_raw << (6'd32 - {1'b0, rot_amt}));
    end

    // --- Instruction Class Detection ---
    wire is_nop        = (i_inst == 32'd0);
    assign is_halt     = (i_inst == 32'hFFFFFFFF);
    wire is_bx         = !is_nop && (i_inst[27:20] == 8'b00010010) && (i_inst[7:4] == 4'b0001);
    wire is_dp         = (op_class == 2'b00) && !is_nop && !is_bx;
    wire is_load_store = (op_class == 2'b01);
    wire is_load       = is_load_store &  L_bit;
    wire is_store      = is_load_store & ~L_bit;
    wire is_test_op    = (opcode[3:2] == 2'b10);
    wire is_branch     = (i_inst[27:25] == 3'b101) && !is_nop;

    // --- Condition Evaluator ---
    reg cond_pass;
    always @(*) begin
        case (cond)
            4'b0000: cond_pass = cpsr_Z;              // EQ
            4'b0001: cond_pass = ~cpsr_Z;             // NE
            4'b0010: cond_pass = cpsr_C;              // CS
            4'b0011: cond_pass = ~cpsr_C;             // CC
            4'b0100: cond_pass = cpsr_N;              // MI
            4'b0101: cond_pass = ~cpsr_N;             // PL
            4'b0110: cond_pass = cpsr_V;              // VS
            4'b0111: cond_pass = ~cpsr_V;             // VC
            4'b1000: cond_pass = cpsr_C & ~cpsr_Z;    // HI
            4'b1001: cond_pass = ~cpsr_C | cpsr_Z;    // LS
            4'b1010: cond_pass = (cpsr_N == cpsr_V);  // GE
            4'b1011: cond_pass = (cpsr_N != cpsr_V);  // LT
            4'b1100: cond_pass = ~cpsr_Z & (cpsr_N == cpsr_V); // GT
            4'b1101: cond_pass = cpsr_Z | (cpsr_N != cpsr_V);  // LE
            4'b1110: cond_pass = 1'b1;                // AL
            4'b1111: cond_pass = 1'b0;
            default: cond_pass = 1'b1;
        endcase
    end

    // --- Branch Target ---
    wire [23:0] branch_imm24    = i_inst[23:0];
    wire [63:0] branch_offset64 = {{38{branch_imm24[23]}}, branch_imm24, 2'b00};
    wire [63:0] branch_target_b = {53'd0, ifid_pc} + 64'd4 + branch_offset64;

    wire [3:0]  bx_rm            = i_inst[3:0];
    wire [63:0] branch_target_bx = register_file[{thread_id, bx_rm}];

    // --- Immediate / Offset MUX ---
    wire [63:0] dp_imm       = {32'd0, imm32};
    wire [63:0] ls_offset    = {52'd0, offset12};
    wire [63:0] imm_extended = is_load_store ? ls_offset : dp_imm;

    // --- Register Read ---
    wire [3:0] r0addr = Rn;
    wire [3:0] r1addr = is_store ? Rd : Rm;

    wire        rn_is_pc     = (Rn == 4'd15);
    wire [63:0] arm_pc_value = {53'd0, ifid_pc} + 64'd4;

    assign r0data = rn_is_pc ? arm_pc_value : register_file[{thread_id, r0addr}];
    assign r1data = register_file[{thread_id, r1addr}];

    // --- Output Assignments ---
    assign o_WReg1         = Rd;
    assign o_sign_extended = imm_extended;
    assign o_shift_amt     = shift_amt;
    assign o_shift_type    = shift_type;
    assign branch_addr     = is_bx ? branch_target_bx : branch_target_b;

    // --- Control Unit ---
    always @(*) begin
        o_WRegEn     = 1'b0;
        o_WMemEn     = 1'b0;
        o_RMemEn     = 1'b0;
        ctrl_mem2reg = 1'b0;
        ctrl_branch  = 1'b0;
        alusrc       = 1'b0;
        ALU_op       = 4'b0100;
        o_set_flags  = 1'b0;

        if (is_branch) begin
            ctrl_branch = cond_pass;
        end
        else if (is_bx) begin
            ctrl_branch = cond_pass;
        end
        else if (is_dp) begin
            ALU_op      = opcode;
            alusrc      = I_bit;
            o_WRegEn    = !is_test_op & cond_pass;
            o_set_flags = (S_bit | is_test_op) & cond_pass;
        end
        else if (is_load) begin
            alusrc       = 1'b1;
            ALU_op       = U_bit ? 4'b0100 : 4'b0010; // ADD or SUB offset
            o_RMemEn     = cond_pass;
            o_WRegEn     = cond_pass;
            ctrl_mem2reg = 1'b1;
        end
        else if (is_store) begin
            alusrc       = 1'b1;
            ALU_op       = U_bit ? 4'b0100 : 4'b0010;
            o_WMemEn     = cond_pass;
        end
    end

endmodule
