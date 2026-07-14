`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_stress_refinements -- Stress tests for refinements.md items
//
// Tests:
//   C1: Conditional execution bug (ADDEQ with Z=0 should NOT execute)
//   C2: 1-word packet hangs FIFO FSM
//   H1: PC not gated by halted flag (runs past HALT)
//   H2: data_mem was_ext has no reset
//   H3: CPU runs when rst=0, start_stop=0
//   M2: Missing default in condition evaluator case

module tb_stress_refinements;

    reg         clk, reset;

    // NF2.1 data path
    reg  [63:0] in_data;
    reg  [7:0]  in_ctrl;
    reg         in_wr;
    wire        in_rdy;

    wire [63:0] out_data;
    wire [7:0]  out_ctrl;
    wire        out_wr;
    reg         out_rdy;

    // Register ring
    reg                              reg_req_in;
    reg                              reg_ack_in;
    reg                              reg_rd_wr_L_in;
    reg  [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_in;
    reg  [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_in;
    reg  [1:0]                       reg_src_in;

    wire                             reg_req_out;
    wire                             reg_ack_out;
    wire                             reg_rd_wr_L_out;
    wire [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_out;
    wire [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_out;
    wire [1:0]                       reg_src_out;

    lab8_wrapper #(
        .DATA_WIDTH(64),
        .CTRL_WIDTH(8),
        .UDP_REG_SRC_WIDTH(2)
    ) uut (
        .clk            (clk),
        .reset          (reset),
        .in_data        (in_data),
        .in_ctrl        (in_ctrl),
        .in_wr          (in_wr),
        .in_rdy         (in_rdy),
        .out_data       (out_data),
        .out_ctrl       (out_ctrl),
        .out_wr         (out_wr),
        .out_rdy        (out_rdy),
        .reg_req_in     (reg_req_in),
        .reg_ack_in     (reg_ack_in),
        .reg_rd_wr_L_in (reg_rd_wr_L_in),
        .reg_addr_in    (reg_addr_in),
        .reg_data_in    (reg_data_in),
        .reg_src_in     (reg_src_in),
        .reg_req_out    (reg_req_out),
        .reg_ack_out    (reg_ack_out),
        .reg_rd_wr_L_out(reg_rd_wr_L_out),
        .reg_addr_out   (reg_addr_out),
        .reg_data_out   (reg_data_out),
        .reg_src_out    (reg_src_out)
    );

    always #5 clk = ~clk;

    function [`UDP_REG_ADDR_WIDTH-1:0] make_addr;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        begin
            make_addr = {`LAB8_BLOCK_ADDR, offset};
        end
    endfunction

    task fifo_write;
        input [7:0]  ctrl;
        input [63:0] data;
        begin
            @(posedge clk); #1;
            in_ctrl = ctrl;
            in_data = data;
            in_wr   = 1'b1;
            @(posedge clk); #1;
            in_wr   = 1'b0;
            in_ctrl = 8'd0;
            in_data = 64'd0;
        end
    endtask

    task pci_write;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            reg_req_in     = 1'b1;
            reg_ack_in     = 1'b0;
            reg_rd_wr_L_in = 1'b0;
            reg_addr_in    = make_addr(offset);
            reg_data_in    = data;
            @(posedge clk); #1;
            reg_req_in     = 1'b0;
            @(posedge clk);
        end
    endtask

    reg [31:0] pci_rd_result;
    task pci_read;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        begin
            @(posedge clk); #1;
            reg_req_in     = 1'b1;
            reg_ack_in     = 1'b0;
            reg_rd_wr_L_in = 1'b1;
            reg_addr_in    = make_addr(offset);
            reg_data_in    = 32'd0;
            @(posedge clk); #1;
            pci_rd_result  = reg_data_out;
            reg_req_in     = 1'b0;
        end
    endtask

    task load_imem;
        input [8:0]  addr;
        input [31:0] data;
        begin
            pci_write(`LAB8_CPU_IMEM_ADDR, {23'd0, addr});
            pci_write(`LAB8_CPU_IMEM_WDATA, data);
            pci_write(`LAB8_CPU_IMEM_CMD, 32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    task soft_reset_fifo;
        begin
            pci_write(`LAB8_CTRL, 32'h1);
            repeat (3) @(posedge clk);
            pci_write(`LAB8_CTRL, 32'h0);
            repeat (2) @(posedge clk);
        end
    endtask

    integer i;
    integer pass;
    integer total_pass;
    integer total_fail;

    // ---------------------------------------------------------------
    // ARM instruction encodings
    // ---------------------------------------------------------------

    // MOV r1, #0x80 (always):   cond=E, op=00, I=1, opcode=1101, S=0, Rn=0, Rd=1, imm=0x80
    localparam [31:0] INST_MOV_R1_128     = 32'hE3A01080;

    // MOV r3, #0 (always):      cond=E, I=1, opcode=1101(MOV), Rd=3, imm=0
    localparam [31:0] INST_MOV_R3_0       = 32'hE3A03000;

    // MOV r4, #100 (always):    cond=E, I=1, opcode=1101(MOV), Rd=4, imm=100(0x64)
    localparam [31:0] INST_MOV_R4_100     = 32'hE3A04064;

    // CMP r3, #0 (sets Z=1):   cond=E, I=1, opcode=1010(CMP), S=1, Rn=3, Rd=0, imm=0
    // Encoding: E3530000
    localparam [31:0] INST_CMP_R3_0       = 32'hE3530000;

    // ADDEQ r4, r4, #1:        cond=0(EQ), I=1, opcode=0100(ADD), S=0, Rn=4, Rd=4, imm=1
    // cond=0000, 00 1 0100 0 0100 0100 000000000001
    // = 0000_0010_1000_0100_0100_0000_0000_0001 = 02844001
    localparam [31:0] INST_ADDEQ_R4_1     = 32'h02844001;

    // ADDNE r4, r4, #1:        cond=1(NE), I=1, opcode=0100(ADD), S=0, Rn=4, Rd=4, imm=1
    // = 1284_4001  -> 0001 00 1 0100 0 0100 0100 000000000001
    localparam [31:0] INST_ADDNE_R4_1     = 32'h12844001;

    // CMP r3, #99 (sets Z=0 since r3=0): E353_0063
    localparam [31:0] INST_CMP_R3_99      = 32'hE3530063;

    // STR r4, [r1, #2]:        cond=E, op=01, U=1, L=0, Rn=1, Rd=4, offset=2
    // = E581_4002
    localparam [31:0] INST_STR_R4_R1_2    = 32'hE5814002;

    // LDR r2, [r1, #1]:        E591_2001
    localparam [31:0] INST_LDR_R2_R1_1    = 32'hE5912001;

    // STR r2, [r1, #1]:        E581_2001
    localparam [31:0] INST_STR_R2_R1_1    = 32'hE5812001;

    // ADD r2, r2, #42:         E282_202A
    localparam [31:0] INST_ADD_R2_42      = 32'hE282202A;

    // NOP: 0x00000000
    localparam [31:0] INST_NOP            = 32'h00000000;

    // HALT: 0xFFFFFFFF
    localparam [31:0] INST_HALT           = 32'hFFFFFFFF;

    // B . (self-branch, offset=-2 words = -8 bytes):
    // cond=E, 101, L=0, offset24 = 0xFFFFFE (-2 in 2's complement 24-bit)
    localparam [31:0] INST_B_SELF         = 32'hEAFFFFFE;

    initial begin
        $dumpfile("tb_stress_refinements.vcd");
        $dumpvars(0, tb_stress_refinements);

        clk = 0; reset = 1;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1;
        reg_req_in = 0; reg_ack_in = 0; reg_rd_wr_L_in = 0;
        reg_addr_in = 0; reg_data_in = 0; reg_src_in = 0;
        total_pass = 0; total_fail = 0;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        // =============================================================
        // TEST C2: 1-word packet hangs FIFO FSM
        // =============================================================
        $display("\n============================================");
        $display("TEST C2: 1-word packet (word_length=1)");
        $display("============================================");
        pass = 1;

        // Inject a packet with word_length=1 (only header, no data words)
        // Header: dst_port=0x0001, word_length=1, src_port=0x0002, byte_length=8
        fifo_write(8'hFF, 64'h0001_0001_0002_0008);

        // Wait some cycles for FSM to process
        repeat (10) @(posedge clk);

        // Check FSM state -- should be in PKT_BUFFERED (state=2) or IDLE
        // BUG: FSM gets stuck in S_RECEIVING (state=1) because
        // words_remaining=0, and exit condition is words_remaining==1
        $display("  FIFO FSM state = %0d (0=IDLE, 1=RECEIVING, 2=PKT_BUFFERED)",
                 uut.u_fifo.state);

        if (uut.u_fifo.state == 3'd1) begin
            $display("  ** BUG CONFIRMED: FSM stuck in S_RECEIVING (state=1)");
            $display("  ** words_remaining = %0d (should have triggered transition)",
                     uut.u_fifo.words_remaining);
            pass = 0;
        end else if (uut.u_fifo.state == 3'd2 || uut.u_fifo.state == 3'd0) begin
            $display("  OK: FSM in expected state");
        end else begin
            $display("  UNEXPECTED: FSM in state %0d", uut.u_fifo.state);
            pass = 0;
        end

        // Also check: can the FIFO accept new packets? (in_rdy should go high after reset)
        // If stuck in RECEIVING, in_rdy stays high but no new packet can complete
        pci_read(`LAB8_STATUS);
        $display("  STATUS = 0x%h (pkt_ready=%b, full=%b, empty=%b)",
                 pci_rd_result, pci_rd_result[0], pci_rd_result[1], pci_rd_result[2]);

        // Try injecting another normal 2-word packet
        fifo_write(8'hFF, 64'h0001_0002_0002_0010); // word_length=2
        fifo_write(8'h00, 64'hDEAD_BEEF_CAFE_BABE); // data word

        repeat (10) @(posedge clk);

        $display("  After 2nd packet: FSM state = %0d", uut.u_fifo.state);
        pci_read(`LAB8_STATUS);
        $display("  STATUS = 0x%h (pkt_ready=%b)", pci_rd_result, pci_rd_result[0]);

        if (uut.u_fifo.state == 3'd1) begin
            $display("  ** BUG CONFIRMED: FSM permanently stuck, 2nd packet also lost");
        end

        if (!pass) begin
            $display("  RESULT: C2 BUG REPRODUCED -- 1-word packet hangs FSM");
            total_fail = total_fail + 1;
        end else begin
            $display("  RESULT: C2 not triggered (FSM handled it)");
            total_pass = total_pass + 1;
        end

        // Reset FIFO for next test
        soft_reset_fifo;

        // =============================================================
        // TEST C1: Conditional execution bug
        // =============================================================
        $display("\n============================================");
        $display("TEST C1: Conditional execution (ADDEQ/ADDNE)");
        $display("============================================");
        pass = 1;

        // Inject a 4-word packet so we have BRAM to work with
        fifo_write(8'hFF, 64'h0001_0004_0002_0000); // header, word_length=4
        fifo_write(8'h00, 64'h0000_0000_0000_0000); // data 1 (all zeros)
        fifo_write(8'h00, 64'h0000_0000_0000_0000); // data 2 (all zeros)
        fifo_write(8'h00, 64'h0000_0000_0000_0000); // data 3 (all zeros)

        repeat (5) @(posedge clk);

        // Switch to CPU mode
        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);

        // Assert CPU reset
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (5) @(posedge clk);

        // Program: Test conditional execution
        //
        //   MOV r3, #0          ; r3 = 0
        //   MOV r4, #100        ; r4 = 100
        //   CMP r3, #99         ; compare 0 vs 99 -> Z=0 (not equal), N=1 (negative)
        //   ADDEQ r4, r4, #1    ; should NOT execute (Z=0, EQ requires Z=1)
        //   MOV r1, #128        ; r1 = 0x80 (FIFO BRAM base)
        //   STR r4, [r1, #2]    ; store r4 to BRAM[2]
        //   HALT
        //
        // Expected: r4 = 100 (ADDEQ skipped)
        // Bug:      r4 = 101 (ADDEQ executes unconditionally)

        for (i = 0; i < 4; i = i + 1) begin
            load_imem(i * 128 + 0,  INST_MOV_R3_0);       // r3 = 0
            load_imem(i * 128 + 1,  INST_MOV_R4_100);     // r4 = 100
            load_imem(i * 128 + 2,  INST_CMP_R3_99);      // CMP r3, #99 -> Z=0
            load_imem(i * 128 + 3,  INST_ADDEQ_R4_1);     // ADDEQ r4, r4, #1 (should skip)
            load_imem(i * 128 + 4,  INST_MOV_R1_128);     // r1 = 0x80
            load_imem(i * 128 + 5,  INST_STR_R4_R1_2);    // STR r4, [r1, #2] -> BRAM[2]
            load_imem(i * 128 + 6,  INST_HALT);
            load_imem(i * 128 + 7,  INST_B_SELF);         // B . safety
        end

        // Start CPU
        pci_write(`LAB8_CPU_CTRL, 32'h1);

        // Wait for completion (generous timeout)
        for (i = 0; i < 400; i = i + 1) begin
            @(posedge clk);
            if (uut.cpu_all_halted) begin
                $display("  CPU halted after %0d cycles", i);
                i = 400;
            end
        end

        // Stop CPU
        pci_write(`LAB8_CPU_CTRL, 32'h0);
        repeat (2) @(posedge clk);

        // Read BRAM[2] to see what r4 was
        pci_write(`LAB8_BRAM_ADDR, 32'd2);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_BRAM_RD_LO);
        $display("  BRAM[2] lo = 0x%h", pci_rd_result);

        if (pci_rd_result == 32'd101) begin
            $display("  ** BUG CONFIRMED: ADDEQ executed when Z=0 (r4=101, should be 100)");
            $display("  ** Conditional execution does NOT gate DP instructions");
            pass = 0;
        end else if (pci_rd_result == 32'd100) begin
            $display("  OK: ADDEQ correctly skipped (r4=100)");
        end else begin
            $display("  UNEXPECTED: r4 = %0d", pci_rd_result);
            pass = 0;
        end

        if (!pass) begin
            $display("  RESULT: C1 BUG REPRODUCED -- conditional execution broken");
            total_fail = total_fail + 1;
        end else begin
            $display("  RESULT: C1 conditional execution works correctly");
            total_pass = total_pass + 1;
        end

        // Reset for next test
        soft_reset_fifo;

        // =============================================================
        // TEST C1b: ADDNE should execute when Z=0 (positive case)
        // =============================================================
        $display("\n============================================");
        $display("TEST C1b: ADDNE should execute when Z=0");
        $display("============================================");
        pass = 1;

        fifo_write(8'hFF, 64'h0001_0004_0002_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        repeat (5) @(posedge clk);

        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (5) @(posedge clk);

        // Program:
        //   MOV r3, #0
        //   MOV r4, #100
        //   CMP r3, #99         ; Z=0 (not equal)
        //   ADDNE r4, r4, #1    ; should execute (Z=0, NE requires Z=0)
        //   MOV r1, #128
        //   STR r4, [r1, #2]
        //   HALT
        //
        // Expected: r4 = 101 (ADDNE executes)
        // Note: with the bug, ADDNE also gives 101 -- but for the wrong reason
        //       (executes unconditionally, not because NE is true)

        for (i = 0; i < 4; i = i + 1) begin
            load_imem(i * 128 + 0,  INST_MOV_R3_0);
            load_imem(i * 128 + 1,  INST_MOV_R4_100);
            load_imem(i * 128 + 2,  INST_CMP_R3_99);
            load_imem(i * 128 + 3,  INST_ADDNE_R4_1);     // ADDNE: should execute
            load_imem(i * 128 + 4,  INST_MOV_R1_128);
            load_imem(i * 128 + 5,  INST_STR_R4_R1_2);
            load_imem(i * 128 + 6,  INST_HALT);
            load_imem(i * 128 + 7,  INST_B_SELF);
        end

        pci_write(`LAB8_CPU_CTRL, 32'h1);
        for (i = 0; i < 400; i = i + 1) begin
            @(posedge clk);
            if (uut.cpu_all_halted) i = 400;
        end
        pci_write(`LAB8_CPU_CTRL, 32'h0);
        repeat (2) @(posedge clk);

        pci_write(`LAB8_BRAM_ADDR, 32'd2);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_BRAM_RD_LO);
        $display("  BRAM[2] lo = 0x%h", pci_rd_result);

        if (pci_rd_result == 32'd101) begin
            $display("  OK: ADDNE executed (r4=101) -- but see C1 to confirm it's not unconditional");
        end else begin
            $display("  UNEXPECTED: r4 = %0d", pci_rd_result);
            pass = 0;
        end

        if (!pass) begin
            total_fail = total_fail + 1;
        end else begin
            total_pass = total_pass + 1;
        end

        soft_reset_fifo;

        // =============================================================
        // TEST H1: PC not gated by halted flag
        // =============================================================
        $display("\n============================================");
        $display("TEST H1: PC keeps advancing after HALT");
        $display("============================================");
        pass = 1;

        fifo_write(8'hFF, 64'h0001_0004_0002_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        repeat (5) @(posedge clk);

        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (5) @(posedge clk);

        // Simple program: just HALT immediately
        // If PC is gated, it should freeze at HALT instruction address.
        // If not gated, PC will advance past HALT.
        for (i = 0; i < 4; i = i + 1) begin
            load_imem(i * 128 + 0, INST_HALT);
            // Fill rest with NOPs so if PC advances, it runs NOPs
            load_imem(i * 128 + 1, INST_NOP);
            load_imem(i * 128 + 2, INST_NOP);
            load_imem(i * 128 + 3, INST_NOP);
        end

        // Start CPU and let it run
        pci_write(`LAB8_CPU_CTRL, 32'h1);

        // Wait enough cycles for all threads to hit HALT and then some
        repeat (50) @(posedge clk);

        // Check: are halted flags set?
        $display("  halted flags = %b", uut.u_cpu.halted);

        // Sample PC for thread 0 over multiple cycles
        // If bug exists: PC keeps changing. If fixed: PC is frozen.
        begin : pc_check_block
            reg [10:0] pc_sample1, pc_sample2, pc_sample3;
            @(posedge clk);
            pc_sample1 = uut.u_cpu.u_if.pc[0];
            repeat (8) @(posedge clk);  // 2 full thread rotations
            pc_sample2 = uut.u_cpu.u_if.pc[0];
            repeat (8) @(posedge clk);
            pc_sample3 = uut.u_cpu.u_if.pc[0];

            $display("  Thread 0 PC samples: %0h, %0h, %0h", pc_sample1, pc_sample2, pc_sample3);

            if (pc_sample1 != pc_sample2 || pc_sample2 != pc_sample3) begin
                $display("  ** BUG CONFIRMED: PC keeps advancing after HALT");
                $display("  ** Thread 0 PC is not frozen (changes between samples)");
                pass = 0;
            end else begin
                $display("  OK: PC frozen after HALT");
            end
        end

        // Also check all_halted
        pci_read(`LAB8_CPU_STATUS);
        $display("  CPU_STATUS = 0x%h (all_halted=%b)", pci_rd_result, pci_rd_result[31]);

        pci_write(`LAB8_CPU_CTRL, 32'h0);

        if (!pass) begin
            $display("  RESULT: H1 BUG REPRODUCED -- PC not gated by halt");
            total_fail = total_fail + 1;
        end else begin
            $display("  RESULT: H1 not present (PC properly gated)");
            total_pass = total_pass + 1;
        end

        soft_reset_fifo;

        // =============================================================
        // TEST H3: CPU runs when rst=0, start_stop=0
        // =============================================================
        $display("\n============================================");
        $display("TEST H3: CPU runs when rst=0, start_stop=0");
        $display("============================================");
        pass = 1;

        fifo_write(8'hFF, 64'h0001_0004_0002_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        fifo_write(8'h00, 64'h0000_0000_0000_0000);
        repeat (5) @(posedge clk);

        // Write a known zero to BRAM[2] via PCI to clear stale data
        pci_write(`LAB8_BRAM_ADDR, 32'd2);
        pci_write(`LAB8_BRAM_WD_LO, 32'd0);
        pci_write(`LAB8_BRAM_WD_HI, 32'd0);
        pci_write(`LAB8_BRAM_WCTRL, 32'd0);
        pci_write(`LAB8_BRAM_CMD, 32'h1);
        repeat (3) @(posedge clk);

        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);

        // Load a program that modifies BRAM[2] to a known value
        // First: hold CPU in reset
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (5) @(posedge clk);

        for (i = 0; i < 4; i = i + 1) begin
            load_imem(i * 128 + 0, INST_MOV_R4_100);
            load_imem(i * 128 + 1, INST_MOV_R1_128);
            load_imem(i * 128 + 2, INST_STR_R4_R1_2);
            load_imem(i * 128 + 3, INST_HALT);
            load_imem(i * 128 + 4, INST_B_SELF);
        end

        // Now: release reset but do NOT set start_stop
        // CPU_CTRL = 0x00 means rst=0, start_stop=0
        pci_write(`LAB8_CPU_CTRL, 32'h0);

        // Wait long enough for the program to potentially execute
        repeat (100) @(posedge clk);

        // Check if BRAM[2] was modified (it shouldn't be if CPU is properly stopped)
        pci_write(`LAB8_BRAM_ADDR, 32'd2);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_BRAM_RD_LO);
        $display("  BRAM[2] lo = 0x%h (should be 0 if CPU stayed stopped)", pci_rd_result);

        if (pci_rd_result == 32'd100) begin
            $display("  ** BUG CONFIRMED: CPU executed program with start_stop=0");
            $display("  ** r4=100 was stored to BRAM[2] even though CPU should be paused");
            pass = 0;
        end else if (pci_rd_result == 32'd0) begin
            $display("  OK: CPU did not execute (BRAM[2] still 0)");
        end else begin
            $display("  UNEXPECTED: BRAM[2] = 0x%h", pci_rd_result);
            pass = 0;
        end

        // Also check: is the thread_id counter advancing?
        begin : tid_check
            reg [1:0] tid1, tid2;
            @(posedge clk); tid1 = uut.u_cpu.thread_id;
            repeat (4) @(posedge clk); tid2 = uut.u_cpu.thread_id;
            if (tid1 == tid2)
                $display("  thread_id counter is frozen (correct when start_stop=0)");
            else begin
                $display("  ** thread_id counter advances with start_stop=0 (tid: %0d -> %0d)",
                         tid1, tid2);
                // Note: thread_id only advances when start_stop=1 per processor.v line 70
            end
        end

        pci_write(`LAB8_CPU_CTRL, 32'h2); // re-assert reset for cleanup

        if (!pass) begin
            $display("  RESULT: H3 BUG REPRODUCED -- CPU runs without start_stop");
            total_fail = total_fail + 1;
        end else begin
            $display("  RESULT: H3 not present (CPU properly gated by start_stop)");
            total_pass = total_pass + 1;
        end

        soft_reset_fifo;

        // =============================================================
        // TEST H2: data_mem was_ext has no reset
        // =============================================================
        $display("\n============================================");
        $display("TEST H2: data_mem was_ext register reset");
        $display("============================================");
        // This is observable by checking the initial value of was_ext
        // In simulation, uninitialized regs are X. On real FPGA, could be 0 or 1.
        // The concern is that the very first MEM stage read could mux wrong source.

        $display("  data_mem was_ext initial value = %b", uut.u_cpu.u_mem.was_ext);
        if (uut.u_cpu.u_mem.was_ext === 1'bx) begin
            $display("  ** ISSUE: was_ext is X (undefined) -- no reset in RTL");
            $display("  ** On FPGA, this means first MEM read could use wrong data source");
            total_fail = total_fail + 1;
        end else begin
            $display("  was_ext has a defined value (sim may default to 0)");
            $display("  NOTE: Still no reset in RTL -- FPGA behavior unpredictable");
            total_pass = total_pass + 1;
        end

        // =============================================================
        // SUMMARY
        // =============================================================
        $display("\n============================================");
        $display("STRESS TEST SUMMARY");
        $display("============================================");
        $display("  Tests passed: %0d", total_pass);
        $display("  Tests with bugs reproduced: %0d", total_fail);
        $display("");
        if (total_fail > 0) begin
            $display("  BUGS CONFIRMED in this run.");
            $display("  See individual test output above for details.");
        end else begin
            $display("  No bugs triggered (check test conditions).");
        end
        $display("============================================\n");

        #100;
        $finish;
    end

endmodule
