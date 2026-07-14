`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_cpu_fifo — Test ARM CPU ↔ FIFO BRAM integration
//
// Tests:
//   1. Inject packet into FIFO, verify buffered
//   2. Switch to CPU mode, load ARM program via PCI
//   3. Start CPU, wait for all_halted
//   4. Verify CPU modified FIFO BRAM word
//   5. Drain, verify modified packet on output
//
// ARM test program (all 4 threads execute same code):
//   MOV r1, #128     ; FIFO base = 0x80 (addr[7]=1 → external FIFO BRAM)
//   LDR r2, [r1, #1] ; load FIFO word 1 (first data word)
//   ADD r2, r2, #42  ; modify: add 42
//   STR r2, [r1, #1] ; store back
//   HALT              ; all threads halt

module tb_cpu_fifo;

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

    // Build full register address from word offset
    function [`UDP_REG_ADDR_WIDTH-1:0] make_addr;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        begin
            make_addr = {`LAB8_BLOCK_ADDR, offset};
        end
    endfunction

    // Task: write one NF2.1 word into data path
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

    // Task: PCI register write
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

    // Task: PCI register read
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

    // ARM instruction encodings
    // MOV r1, #128:  E3A01080
    // LDR r2, [r1, #1]:  E5912001
    // ADD r2, r2, #42:  E282202A
    // STR r2, [r1, #1]:  E5812001
    // HALT:  FFFFFFFF
    localparam [31:0] INST_MOV_R1_128  = 32'hE3A01080;
    localparam [31:0] INST_LDR_R2_R1_1 = 32'hE5912001;
    localparam [31:0] INST_ADD_R2_42   = 32'hE282202A;
    localparam [31:0] INST_STR_R2_R1_1 = 32'hE5812001;
    localparam [31:0] INST_HALT        = 32'hFFFFFFFF;

    // Task: load one IMEM word via PCI
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

    // Collected output
    reg [71:0] captured_out [0:31];
    integer    out_idx;
    integer    i;
    integer    pass;

    initial begin
        $dumpfile("tb_cpu_fifo.vcd");
        $dumpvars(0, tb_cpu_fifo);

        // Init
        clk = 0; reset = 1;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1;
        reg_req_in = 0; reg_ack_in = 0; reg_rd_wr_L_in = 0;
        reg_addr_in = 0; reg_data_in = 0; reg_src_in = 0;
        out_idx = 0; pass = 1;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        // =====================================================
        $display("\n=== Test 1: Inject 4-word packet ===");

        fifo_write(8'hFF, 64'h0001_0004_0018_0000); // header: word_length=4
        fifo_write(8'h00, 64'hAAAA_BBBB_CCCC_DDDD); // data 1
        fifo_write(8'h00, 64'h1111_2222_3333_4444); // data 2
        fifo_write(8'h00, 64'h5555_6666_7777_8888); // data 3

        repeat (3) @(posedge clk);

        pci_read(`LAB8_STATUS);
        if (pci_rd_result[0] !== 1'b1) begin
            $display("FAIL: pkt_ready not set");
            pass = 0;
        end else
            $display("PASS: pkt_ready set");

        // =====================================================
        $display("\n=== Test 2: Load ARM program, reset CPU ===");

        // Switch to CPU mode
        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);

        // Assert CPU reset (bit[1])
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (5) @(posedge clk);

        // Load program into IMEM for all 4 thread regions
        // Thread N region starts at IMEM word index N*128
        // (IF_stage: PC = {thread_id, offset[8:0]}, word index = PC[10:2])
        for (i = 0; i < 4; i = i + 1) begin
            load_imem(i * 128 + 0, INST_MOV_R1_128);
            load_imem(i * 128 + 1, INST_LDR_R2_R1_1);
            load_imem(i * 128 + 2, INST_ADD_R2_42);
            load_imem(i * 128 + 3, INST_STR_R2_R1_1);
            load_imem(i * 128 + 4, INST_HALT);
        end

        // Read back IMEM[0] to verify
        pci_write(`LAB8_CPU_IMEM_ADDR, 32'd0);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_CPU_IMEM_RDATA);
        if (pci_rd_result !== INST_MOV_R1_128) begin
            $display("FAIL: IMEM[0] readback = 0x%h, expected 0x%h",
                     pci_rd_result, INST_MOV_R1_128);
            pass = 0;
        end else
            $display("PASS: IMEM[0] loaded correctly");

        // =====================================================
        $display("\n=== Test 3: Start CPU, wait for halt ===");

        // Release CPU reset, set start_stop=1 (bits: [1]=0 reset off, [0]=1 run)
        pci_write(`LAB8_CPU_CTRL, 32'h1);

        // Wait for all_halted (CPU_STATUS bit[31])
        // 4 threads × 5 instructions × 4 cycles/thread = ~80 cycles max
        for (i = 0; i < 200; i = i + 1) begin
            @(posedge clk);
            if (uut.cpu_all_halted) begin
                $display("CPU halted after %0d cycles", i);
                i = 200;
            end
        end

        pci_read(`LAB8_CPU_STATUS);
        $display("CPU_STATUS = 0x%h", pci_rd_result);
        if (pci_rd_result[31] !== 1'b1) begin
            $display("FAIL: all_halted not set in CPU_STATUS");
            pass = 0;
        end else
            $display("PASS: all threads halted");

        // Stop CPU
        pci_write(`LAB8_CPU_CTRL, 32'h0);
        repeat (2) @(posedge clk);

        // =====================================================
        $display("\n=== Test 4: Verify FIFO BRAM modification ===");

        // Read FIFO BRAM word 1 via PCI (CPU should have modified it)
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        repeat (2) @(posedge clk);

        pci_read(`LAB8_BRAM_RD_LO);
        $display("BRAM[1] data_lo = 0x%h", pci_rd_result);

        pci_read(`LAB8_BRAM_RD_HI);
        $display("BRAM[1] data_hi = 0x%h", pci_rd_result);

        // Original data word 1: 64'hAAAA_BBBB_CCCC_DDDD
        // CPU does ADD r2, r2, #42 (42 = 0x2A)
        // All 4 threads read same original value, store val+42
        // Final value: 0xAAAA_BBBB_CCCC_DDDD + 42 = 0xAAAA_BBBB_CCCC_DE07
        // (lower 32 bits: 0xCCCC_DDDD + 0x2A = 0xCCCC_DE07)
        // Wait — 0xDDDD + 0x2A = 0xDE07. Yes.
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_BRAM_RD_LO);
        if (pci_rd_result !== 32'hCCCC_DE07) begin
            $display("FAIL: BRAM[1] lo = 0x%h, expected 0xCCCCDE07", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: CPU modified BRAM[1] lo correctly");

        pci_read(`LAB8_BRAM_RD_HI);
        if (pci_rd_result !== 32'hAAAA_BBBB) begin
            $display("FAIL: BRAM[1] hi = 0x%h, expected 0xAAAABBBB", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: BRAM[1] hi unchanged (no carry)");

        // =====================================================
        $display("\n=== Test 5: Drain and verify output ===");

        // Switch back to FIFO mode
        pci_write(`LAB8_FIFO_MODE, 32'd0);
        repeat (2) @(posedge clk);

        // Trigger drain
        pci_write(`LAB8_FIFO_DRAIN, 32'h1);

        // Capture output
        out_idx = 0;
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (out_wr && out_rdy) begin
                captured_out[out_idx] = {out_ctrl, out_data};
                $display("  OUT[%0d]: ctrl=%h data=%h", out_idx, out_ctrl, out_data);
                out_idx = out_idx + 1;
            end
            if (uut.u_fifo.fifo_empty) begin
                i = 100;
            end
        end

        $display("Drained %0d words", out_idx);
        if (out_idx !== 4) begin
            $display("FAIL: expected 4 output words, got %0d", out_idx);
            pass = 0;
        end else
            $display("PASS: 4 words drained");

        // Verify modified word
        if (captured_out[1][63:0] !== 64'hAAAA_BBBB_CCCC_DE07) begin
            $display("FAIL: output word 1 = %h, expected AAAABBBBCCCCDE07",
                     captured_out[1][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 1 shows CPU modification (+42)");

        // Verify unmodified word
        if (captured_out[2][63:0] !== 64'h1111_2222_3333_4444) begin
            $display("FAIL: output word 2 corrupted: %h", captured_out[2][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 2 unchanged");

        // =====================================================
        $display("\n========================================");
        if (pass)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("========================================\n");

        $finish;
    end

endmodule
