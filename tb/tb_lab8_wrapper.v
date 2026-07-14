`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_lab8_wrapper — Test lab8_wrapper (FIFO + PCI register interface)
//
// Tests:
//   1. Inject packet via NF2.1 data path, verify status via register read
//   2. Switch to CPU mode, read BRAM via PCI registers
//   3. Modify BRAM word via PCI registers
//   4. Switch back to FIFO mode, drain, verify output
//   5. Second packet (reuse)

module tb_lab8_wrapper;

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
    // reg_addr = {BLOCK_ADDR[16:0], word_offset[5:0]} = 23 bits
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

    // Task: PCI register write (via register ring)
    // Sets up ring signals, waits for wrapper to latch and process
    task pci_write;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            reg_req_in     = 1'b1;
            reg_ack_in     = 1'b0;
            reg_rd_wr_L_in = 1'b0;   // write
            reg_addr_in    = make_addr(offset);
            reg_data_in    = data;
            @(posedge clk); #1;
            reg_req_in     = 1'b0;
            @(posedge clk); // let pipeline settle
        end
    endtask

    // Task: PCI register read (via register ring)
    // The wrapper latches the request and outputs data 1 cycle later.
    reg [31:0] pci_rd_result;

    task pci_read;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        begin
            @(posedge clk); #1;
            reg_req_in     = 1'b1;
            reg_ack_in     = 1'b0;
            reg_rd_wr_L_in = 1'b1;   // read
            reg_addr_in    = make_addr(offset);
            reg_data_in    = 32'd0;
            @(posedge clk); #1;       // wrapper latches & processes
            pci_rd_result  = reg_data_out;  // capture result immediately
            reg_req_in     = 1'b0;    // then clear request
        end
    endtask

    // Collected output
    reg [71:0] captured_out [0:31];
    integer    out_idx;
    integer    i;
    integer    pass;

    initial begin
        $dumpfile("tb_lab8_wrapper.vcd");
        $dumpvars(0, tb_lab8_wrapper);

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
        $display("\n=== Test 1: Inject 4-word packet, check status via PCI ===");

        fifo_write(8'hFF, 64'h0001_0004_0018_0000); // header: word_length=4
        fifo_write(8'h00, 64'hAAAA_BBBB_CCCC_DDDD); // data 1
        fifo_write(8'h00, 64'h1111_2222_3333_4444); // data 2
        fifo_write(8'h00, 64'h5555_6666_7777_8888); // data 3

        repeat (3) @(posedge clk);

        // Read STATUS register via PCI
        pci_read(`LAB8_STATUS);
        $display("STATUS = 0x%h", pci_rd_result);
        if (pci_rd_result[0] !== 1'b1) begin
            $display("FAIL: pkt_ready not set in STATUS");
            pass = 0;
        end else
            $display("PASS: pkt_ready set in STATUS");

        if (pci_rd_result[1] !== 1'b1) begin
            $display("FAIL: fifo_full not set in STATUS");
            pass = 0;
        end else
            $display("PASS: fifo_full set in STATUS");

        // Check pkt_word_cnt in STATUS[31:24]
        if (pci_rd_result[31:24] !== 8'd4) begin
            $display("FAIL: pkt_word_cnt = %0d, expected 4", pci_rd_result[31:24]);
            pass = 0;
        end else
            $display("PASS: pkt_word_cnt = 4");

        // =====================================================
        $display("\n=== Test 2: PCI BRAM read (CPU mode) ===");

        // Switch to CPU mode
        pci_write(`LAB8_FIFO_MODE, 32'd1);
        repeat (2) @(posedge clk);

        // Read BRAM word 0 (header): set addr, wait, read back
        pci_write(`LAB8_BRAM_ADDR, 32'd0);
        repeat (2) @(posedge clk); // wait for BRAM latency

        pci_read(`LAB8_BRAM_RCTRL);
        $display("BRAM[0] ctrl = 0x%h", pci_rd_result[7:0]);
        if (pci_rd_result[7:0] !== 8'hFF) begin
            $display("FAIL: header ctrl != 0xFF");
            pass = 0;
        end else
            $display("PASS: header ctrl = 0xFF");

        pci_read(`LAB8_BRAM_RD_LO);
        $display("BRAM[0] data_lo = 0x%h", pci_rd_result);

        // Read BRAM word 1 (first data word)
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        repeat (2) @(posedge clk);

        pci_read(`LAB8_BRAM_RD_LO);
        if (pci_rd_result !== 32'hCCCC_DDDD) begin
            $display("FAIL: BRAM[1] data_lo = 0x%h, expected 0xCCCCDDDD", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: BRAM[1] data_lo matches");

        pci_read(`LAB8_BRAM_RD_HI);
        if (pci_rd_result !== 32'hAAAA_BBBB) begin
            $display("FAIL: BRAM[1] data_hi = 0x%h, expected 0xAAAABBBB", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: BRAM[1] data_hi matches");

        // =====================================================
        $display("\n=== Test 3: PCI BRAM write (modify word 1) ===");

        // Write new data to BRAM[1]
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        pci_write(`LAB8_BRAM_WD_LO, 32'h9ABC_DEF0);
        pci_write(`LAB8_BRAM_WD_HI, 32'h1234_5678);
        pci_write(`LAB8_BRAM_WCTRL, 32'h00);
        pci_write(`LAB8_BRAM_CMD, 32'h1);  // write strobe
        repeat (3) @(posedge clk);

        // Read back to verify
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        repeat (2) @(posedge clk);

        pci_read(`LAB8_BRAM_RD_LO);
        if (pci_rd_result !== 32'h9ABC_DEF0) begin
            $display("FAIL: modified BRAM[1] lo = 0x%h", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: modified BRAM[1] lo verified");

        pci_read(`LAB8_BRAM_RD_HI);
        if (pci_rd_result !== 32'h1234_5678) begin
            $display("FAIL: modified BRAM[1] hi = 0x%h", pci_rd_result);
            pass = 0;
        end else
            $display("PASS: modified BRAM[1] hi verified");

        // =====================================================
        $display("\n=== Test 4: Drain via PCI ===");

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
        if (captured_out[1][63:0] !== 64'h1234_5678_9ABC_DEF0) begin
            $display("FAIL: output word 1 not modified: %h", captured_out[1][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 1 shows PCI modification");

        // Verify unmodified word
        if (captured_out[2][63:0] !== 64'h1111_2222_3333_4444) begin
            $display("FAIL: output word 2 corrupted: %h", captured_out[2][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 2 unchanged");

        // =====================================================
        $display("\n=== Test 5: Verify IDLE, second packet ===");
        repeat (5) @(posedge clk);

        // Check status
        pci_read(`LAB8_STATUS);
        if (pci_rd_result[2] !== 1'b1) begin
            $display("FAIL: fifo_empty not set after drain");
            pass = 0;
        end else
            $display("PASS: fifo_empty after drain");

        if (pci_rd_result[0] !== 1'b0) begin
            $display("FAIL: pkt_ready still set");
            pass = 0;
        end else
            $display("PASS: pkt_ready cleared");

        // Second packet
        fifo_write(8'hFF, 64'h0002_0003_0010_0000); // 3-word packet
        fifo_write(8'h00, 64'hDEAD_DEAD_DEAD_DEAD);
        fifo_write(8'h00, 64'hBEEF_BEEF_BEEF_BEEF);

        repeat (3) @(posedge clk);

        pci_read(`LAB8_STATUS);
        if (pci_rd_result[0] !== 1'b1) begin
            $display("FAIL: 2nd packet pkt_ready not set");
            pass = 0;
        end else
            $display("PASS: 2nd packet buffered");

        // Drain second packet
        pci_write(`LAB8_FIFO_DRAIN, 32'h1);
        out_idx = 0;
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (out_wr && out_rdy) begin
                captured_out[out_idx] = {out_ctrl, out_data};
                out_idx = out_idx + 1;
            end
            if (uut.u_fifo.fifo_empty) i = 50;
        end

        if (out_idx !== 3) begin
            $display("FAIL: 2nd packet: expected 3 words, got %0d", out_idx);
            pass = 0;
        end else
            $display("PASS: 2nd packet drained correctly (%0d words)", out_idx);

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
