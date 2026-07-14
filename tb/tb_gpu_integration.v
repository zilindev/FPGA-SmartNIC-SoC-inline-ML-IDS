`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_gpu_integration — Test GPU ↔ FIFO DMA integration
//
// Test 1: DMA round-trip (HALT-only GPU, data unchanged)
//   Inject packet → DMA FIFO→GPU → GPU HALT → DMA GPU→FIFO → drain
//   Verify output matches input
//
// Test 2: GPU modifies data
//   Inject packet → DMA FIFO→GPU → GPU adds 42 to DMEM[0] → DMA GPU→FIFO → drain
//   Verify first data word modified, rest unchanged

module tb_gpu_integration;

    reg         clk, reset;

    reg  [63:0] in_data;
    reg  [7:0]  in_ctrl;
    reg         in_wr;
    wire        in_rdy;

    wire [63:0] out_data;
    wire [7:0]  out_ctrl;
    wire        out_wr;
    reg         out_rdy;

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

    // Task: load one GPU IMEM word via PCI
    task gpu_load_imem;
        input [9:0]  addr;
        input [31:0] data;
        begin
            pci_write(`LAB8_GPU_IMEM_ADDR, {22'd0, addr});
            pci_write(`LAB8_GPU_IMEM_WDATA, data);
            pci_write(`LAB8_GPU_IMEM_CMD, 32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    // Task: start DMA and wait for completion
    task dma_transfer;
        input [7:0]  fifo_addr;
        input [9:0]  gpu_addr;
        input [7:0]  length;
        input        direction;  // 0=FIFO→GPU, 1=GPU→FIFO
        integer dma_wait;
        begin
            pci_write(`LAB8_DMA_FIFO_ADDR, {24'd0, fifo_addr});
            pci_write(`LAB8_DMA_GPU_ADDR, {22'd0, gpu_addr});
            pci_write(`LAB8_DMA_LENGTH, {24'd0, length});
            pci_write(`LAB8_DMA_CTRL, {30'd0, direction, 1'b1});
            // Wait for DMA completion
            for (dma_wait = 0; dma_wait < 500; dma_wait = dma_wait + 1) begin
                @(posedge clk);
                if (!uut.dma_active) begin
                    dma_wait = 500;
                end
            end
            repeat (2) @(posedge clk);
        end
    endtask

    // Packet data constants
    localparam [63:0] PKT_HEADER = 64'h0001_0005_0028_0000; // word_length=5
    localparam [63:0] PKT_DATA0  = 64'h0001_0002_0003_0004;
    localparam [63:0] PKT_DATA1  = 64'hAAAA_BBBB_CCCC_DDDD;
    localparam [63:0] PKT_DATA2  = 64'h1111_2222_3333_4444;
    localparam [63:0] PKT_DATA3  = 64'h5555_6666_7777_8888;

    // GPU instruction encodings (see gpu_params.vh for ISA)
    // HALT: opcode=11111, rest zeros
    localparam [31:0] GPU_HALT = 32'hF800_0000;

    // Collected output
    reg [71:0] captured_out [0:31];
    integer    out_idx;
    integer    i;
    integer    pass;

    initial begin
        $dumpfile("tb_gpu_integration.vcd");
        $dumpvars(0, tb_gpu_integration);

        clk = 0; reset = 1;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1;
        reg_req_in = 0; reg_ack_in = 0; reg_rd_wr_L_in = 0;
        reg_addr_in = 0; reg_data_in = 0; reg_src_in = 0;
        out_idx = 0; pass = 1;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        // =============================================================
        // TEST 1: DMA round-trip with HALT-only GPU
        // =============================================================
        $display("\n=== Test 1: DMA round-trip (GPU HALT, data passthrough) ===");

        // 1a. Inject 5-word packet
        $display("  Injecting 5-word packet...");
        fifo_write(8'hFF, PKT_HEADER);
        fifo_write(8'h00, PKT_DATA0);
        fifo_write(8'h00, PKT_DATA1);
        fifo_write(8'h00, PKT_DATA2);
        fifo_write(8'h00, PKT_DATA3);

        repeat (3) @(posedge clk);

        pci_read(`LAB8_STATUS);
        if (pci_rd_result[0] !== 1'b1) begin
            $display("  FAIL: pkt_ready not set (status=0x%h)", pci_rd_result);
            pass = 0;
        end else
            $display("  PASS: pkt_ready set");

        // 1b. Switch to GPU mode
        pci_write(`LAB8_FIFO_MODE, 32'd2);
        repeat (2) @(posedge clk);

        // 1c. GPU is already in reset (default). Load HALT into IMEM[0].
        $display("  Loading GPU IMEM[0] = HALT...");
        gpu_load_imem(10'd0, GPU_HALT);

        // 1d. DMA FIFO→GPU: copy FIFO[1..4] to GPU DMEM[0..3]
        $display("  DMA FIFO→GPU: FIFO[1..4] → GPU DMEM[0..3]...");
        dma_transfer(8'd1, 10'd0, 8'd4, 1'b0);

        // 1e. Verify GPU DMEM via PCI read
        pci_write(`LAB8_GPU_DMEM_ADDR, 32'd0);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_GPU_DMEM_RDATA_LO);
        if (pci_rd_result !== PKT_DATA0[31:0]) begin
            $display("  FAIL: GPU DMEM[0] lo = 0x%h, expected 0x%h",
                     pci_rd_result, PKT_DATA0[31:0]);
            pass = 0;
        end else
            $display("  PASS: GPU DMEM[0] lo matches FIFO data");

        pci_read(`LAB8_GPU_DMEM_RDATA_HI);
        if (pci_rd_result !== PKT_DATA0[63:32]) begin
            $display("  FAIL: GPU DMEM[0] hi = 0x%h, expected 0x%h",
                     pci_rd_result, PKT_DATA0[63:32]);
            pass = 0;
        end else
            $display("  PASS: GPU DMEM[0] hi matches FIFO data");

        // Check DMEM[1] too
        pci_write(`LAB8_GPU_DMEM_ADDR, 32'd1);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_GPU_DMEM_RDATA_LO);
        if (pci_rd_result !== PKT_DATA1[31:0]) begin
            $display("  FAIL: GPU DMEM[1] lo = 0x%h, expected 0x%h",
                     pci_rd_result, PKT_DATA1[31:0]);
            pass = 0;
        end else
            $display("  PASS: GPU DMEM[1] lo matches FIFO data");

        // 1f. Release GPU reset → hits HALT immediately
        $display("  Releasing GPU reset...");
        pci_write(`LAB8_GPU_CTRL, 32'd0);

        // Wait for kernel_done
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (uut.gpu_kernel_done) begin
                $display("  GPU halted after %0d cycles", i);
                i = 50;
            end
        end

        pci_read(`LAB8_GPU_STATUS);
        if (pci_rd_result[0] !== 1'b1) begin
            $display("  FAIL: kernel_done not set (GPU_STATUS=0x%h)", pci_rd_result);
            pass = 0;
        end else
            $display("  PASS: kernel_done set");

        // Re-assert GPU reset before DMA back
        pci_write(`LAB8_GPU_CTRL, 32'h1);
        repeat (2) @(posedge clk);

        // 1g. DMA GPU→FIFO: copy GPU DMEM[0..3] back to FIFO[1..4]
        $display("  DMA GPU→FIFO: GPU DMEM[0..3] → FIFO[1..4]...");
        dma_transfer(8'd1, 10'd0, 8'd4, 1'b1);

        // 1h. Verify FIFO BRAM[1] via PCI
        pci_write(`LAB8_BRAM_ADDR, 32'd1);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_BRAM_RD_LO);
        if (pci_rd_result !== PKT_DATA0[31:0]) begin
            $display("  FAIL: FIFO BRAM[1] lo = 0x%h, expected 0x%h",
                     pci_rd_result, PKT_DATA0[31:0]);
            pass = 0;
        end else
            $display("  PASS: FIFO BRAM[1] lo intact after round-trip");

        // 1i. Drain and verify
        $display("  Draining...");
        pci_write(`LAB8_FIFO_MODE, 32'd0);
        repeat (2) @(posedge clk);
        pci_write(`LAB8_FIFO_DRAIN, 32'h1);

        out_idx = 0;
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (out_wr && out_rdy) begin
                captured_out[out_idx] = {out_ctrl, out_data};
                out_idx = out_idx + 1;
            end
            if (uut.u_fifo.fifo_empty)
                i = 100;
        end

        $display("  Drained %0d words", out_idx);
        if (out_idx !== 5) begin
            $display("  FAIL: expected 5 output words, got %0d", out_idx);
            pass = 0;
        end else
            $display("  PASS: 5 words drained");

        // Header preserved
        if (captured_out[0][71:64] !== 8'hFF) begin
            $display("  FAIL: header ctrl = 0x%h, expected 0xFF", captured_out[0][71:64]);
            pass = 0;
        end else
            $display("  PASS: header ctrl preserved (0xFF)");

        // Data words unchanged
        if (captured_out[1][63:0] !== PKT_DATA0) begin
            $display("  FAIL: out[1] = 0x%h, expected 0x%h", captured_out[1][63:0], PKT_DATA0);
            pass = 0;
        end else
            $display("  PASS: data word 0 unchanged after round-trip");

        if (captured_out[4][63:0] !== PKT_DATA3) begin
            $display("  FAIL: out[4] = 0x%h, expected 0x%h", captured_out[4][63:0], PKT_DATA3);
            pass = 0;
        end else
            $display("  PASS: data word 3 unchanged after round-trip");

        // =============================================================
        // SUMMARY
        // =============================================================
        $display("\n========================================");
        if (pass)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("========================================\n");

        $finish;
    end

endmodule
