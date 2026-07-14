`timescale 1ns / 1ps

// tb_convertible_fifo — Testbench for convertible FIFO
//
// Test cases:
//   1. Inject a packet, verify capture and pkt_ready
//   2. Read back BRAM contents in CPU mode
//   3. Modify a word in CPU mode
//   4. Drain packet, verify output matches (with modification)
//   5. Verify FIFO returns to IDLE, ready for next packet

module tb_convertible_fifo;

    reg         clk, reset;
    reg  [63:0] in_data;
    reg  [7:0]  in_ctrl;
    reg         in_wr;
    wire        in_rdy;

    wire [63:0] out_data;
    wire [7:0]  out_ctrl;
    wire        out_wr;
    reg         out_rdy;

    reg  [7:0]  proc_addr;
    reg  [71:0] proc_din;
    reg         proc_we;
    wire [71:0] proc_dout;

    reg  [1:0]  mode;
    reg         drain_start;

    wire        pkt_ready;
    wire        fifo_full;
    wire        fifo_empty;
    wire [7:0]  pkt_word_cnt;

    reg  [7:0]  head_wr_val, tail_wr_val;
    reg         head_wr_en, tail_wr_en;
    wire [7:0]  head_rd, tail_rd;

    convertible_fifo uut (
        .clk          (clk),
        .reset        (reset),
        .in_data      (in_data),
        .in_ctrl      (in_ctrl),
        .in_wr        (in_wr),
        .in_rdy       (in_rdy),
        .out_data     (out_data),
        .out_ctrl     (out_ctrl),
        .out_wr       (out_wr),
        .out_rdy      (out_rdy),
        .proc_addr    (proc_addr),
        .proc_din     (proc_din),
        .proc_we      (proc_we),
        .proc_dout    (proc_dout),
        .mode         (mode),
        .drain_start  (drain_start),
        .pkt_ready    (pkt_ready),
        .fifo_full    (fifo_full),
        .fifo_empty   (fifo_empty),
        .pkt_word_cnt (pkt_word_cnt),
        .head_wr_val  (head_wr_val),
        .head_wr_en   (head_wr_en),
        .tail_wr_val  (tail_wr_val),
        .tail_wr_en   (tail_wr_en),
        .head_rd      (head_rd),
        .tail_rd      (tail_rd)
    );

    always #5 clk = ~clk;

    // Task: write one NF2.1 word into FIFO
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

    // Task: read BRAM via processor port (1-cycle latency)
    task proc_read;
        input [7:0] addr;
        begin
            @(posedge clk); #1;
            proc_addr = addr;
            proc_we   = 1'b0;
            @(posedge clk); // BRAM latency
            @(posedge clk); // data now valid on proc_dout
        end
    endtask

    // Task: write BRAM via processor port
    task proc_write;
        input [7:0]  addr;
        input [71:0] data;
        begin
            @(posedge clk); #1;
            proc_addr = addr;
            proc_din  = data;
            proc_we   = 1'b1;
            @(posedge clk); #1;
            proc_we   = 1'b0;
        end
    endtask

    // Collected output
    reg [71:0] captured_out [0:31];
    integer    out_idx;

    integer i;
    integer pass;

    initial begin
        $dumpfile("tb_convertible_fifo.vcd");
        $dumpvars(0, tb_convertible_fifo);

        // Init
        clk = 0; reset = 1;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1;
        proc_addr = 0; proc_din = 0; proc_we = 0;
        mode = 2'd0; drain_start = 0;
        head_wr_val = 0; head_wr_en = 0;
        tail_wr_val = 0; tail_wr_en = 0;
        out_idx = 0; pass = 1;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        // =====================================================
        $display("\n=== Test 1: Inject 6-word packet ===");
        // Header word: ctrl=0xFF, data has word_length=6 in bits[31:16]
        //   data = {src_port=16'h0001, word_length=16'h0006, byte_length=16'h0028}
        //        = 64'h0001_0006_0028_0000
        // (byte_length = 40 = 5 data words * 8 bytes)

        fifo_write(8'hFF, 64'h0001_0006_0028_0000); // header (word_length=6)
        fifo_write(8'h00, 64'hDEAD_BEEF_CAFE_BABE); // data word 1
        fifo_write(8'h00, 64'h1111_2222_3333_4444); // data word 2
        fifo_write(8'h00, 64'h5555_6666_7777_8888); // data word 3
        fifo_write(8'h00, 64'hAAAA_BBBB_CCCC_DDDD); // data word 4
        fifo_write(8'h00, 64'hEEEE_FFFF_0000_1111); // data word 5

        repeat (3) @(posedge clk);

        if (pkt_ready !== 1'b1) begin
            $display("FAIL: pkt_ready not asserted after packet injection");
            pass = 0;
        end else
            $display("PASS: pkt_ready asserted");

        if (fifo_full !== 1'b1) begin
            $display("FAIL: fifo_full not asserted");
            pass = 0;
        end else
            $display("PASS: fifo_full asserted");

        if (pkt_word_cnt !== 8'd6) begin
            $display("FAIL: pkt_word_cnt = %0d, expected 6", pkt_word_cnt);
            pass = 0;
        end else
            $display("PASS: pkt_word_cnt = 6");

        // =====================================================
        $display("\n=== Test 2: CPU mode — read BRAM contents ===");
        mode = 2'd1; // CPU mode
        repeat (2) @(posedge clk);

        // Read word 0 (header)
        proc_read(8'd0);
        $display("BRAM[0] = %h (ctrl=%h)", proc_dout[63:0], proc_dout[71:64]);
        if (proc_dout[71:64] !== 8'hFF) begin
            $display("FAIL: header ctrl != 0xFF");
            pass = 0;
        end else
            $display("PASS: header ctrl = 0xFF");

        // Read word 1 (first data)
        proc_read(8'd1);
        $display("BRAM[1] = %h (ctrl=%h)", proc_dout[63:0], proc_dout[71:64]);
        if (proc_dout[63:0] !== 64'hDEAD_BEEF_CAFE_BABE) begin
            $display("FAIL: data word 1 mismatch");
            pass = 0;
        end else
            $display("PASS: data word 1 matches");

        // =====================================================
        $display("\n=== Test 3: CPU mode — modify word 1 ===");
        proc_write(8'd1, {8'h00, 64'h1234_5678_9ABC_DEF0});
        repeat (2) @(posedge clk);

        // Read back to verify
        proc_read(8'd1);
        if (proc_dout[63:0] !== 64'h1234_5678_9ABC_DEF0) begin
            $display("FAIL: modified word readback mismatch: %h", proc_dout[63:0]);
            pass = 0;
        end else
            $display("PASS: modified word verified");

        // =====================================================
        $display("\n=== Test 4: Drain packet ===");
        mode = 2'd0; // back to FIFO mode
        out_idx = 0;

        repeat (2) @(posedge clk);
        #1 drain_start = 1'b1;
        @(posedge clk);
        #1 drain_start = 1'b0;

        // Capture output words
        // Wait for drain to complete (timeout after 100 cycles)
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (out_wr && out_rdy) begin
                captured_out[out_idx] = {out_ctrl, out_data};
                $display("  OUT[%0d]: ctrl=%h data=%h", out_idx, out_ctrl, out_data);
                out_idx = out_idx + 1;
            end
            if (fifo_empty) begin
                i = 100; // break
            end
        end

        $display("Drained %0d words", out_idx);
        if (out_idx !== 6) begin
            $display("FAIL: expected 6 output words, got %0d", out_idx);
            pass = 0;
        end else
            $display("PASS: 6 words drained");

        // Verify header
        if (captured_out[0][71:64] !== 8'hFF) begin
            $display("FAIL: output header ctrl != 0xFF");
            pass = 0;
        end else
            $display("PASS: output header ctrl correct");

        // Verify modified word
        if (captured_out[1][63:0] !== 64'h1234_5678_9ABC_DEF0) begin
            $display("FAIL: output word 1 not modified: %h", captured_out[1][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 1 shows CPU modification");

        // Verify unmodified word 2
        if (captured_out[2][63:0] !== 64'h1111_2222_3333_4444) begin
            $display("FAIL: output word 2 corrupted: %h", captured_out[2][63:0]);
            pass = 0;
        end else
            $display("PASS: output word 2 unchanged");

        // =====================================================
        $display("\n=== Test 5: Verify FIFO returns to IDLE ===");
        repeat (5) @(posedge clk);

        if (fifo_empty !== 1'b1) begin
            $display("FAIL: fifo_empty not asserted after drain");
            pass = 0;
        end else
            $display("PASS: fifo_empty after drain");

        if (pkt_ready !== 1'b0) begin
            $display("FAIL: pkt_ready still asserted");
            pass = 0;
        end else
            $display("PASS: pkt_ready cleared");

        if (in_rdy !== 1'b1) begin
            $display("FAIL: in_rdy not re-asserted");
            pass = 0;
        end else
            $display("PASS: in_rdy ready for next packet");

        // =====================================================
        $display("\n=== Test 6: Second packet (verify reuse) ===");
        fifo_write(8'hFF, 64'h0002_0003_0010_0000); // 3-word packet
        fifo_write(8'h00, 64'hAAAA_AAAA_AAAA_AAAA);
        fifo_write(8'h00, 64'hBBBB_BBBB_BBBB_BBBB);

        repeat (3) @(posedge clk);

        if (pkt_ready !== 1'b1) begin
            $display("FAIL: pkt_ready not asserted for 2nd packet");
            pass = 0;
        end else
            $display("PASS: 2nd packet buffered");

        // Drain without modification
        #1 drain_start = 1'b1;
        @(posedge clk);
        #1 drain_start = 1'b0;

        out_idx = 0;
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (out_wr && out_rdy) begin
                captured_out[out_idx] = {out_ctrl, out_data};
                out_idx = out_idx + 1;
            end
            if (fifo_empty) i = 50;
        end

        if (out_idx !== 3) begin
            $display("FAIL: 2nd packet drain: expected 3 words, got %0d", out_idx);
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
