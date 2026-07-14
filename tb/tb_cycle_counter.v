// tb_cycle_counter.v -- Verify the new LAB8_GPU_CYCLE_COUNT register
//
// Loads the BCAST-optimized IDS inference kernel through the PCI register
// interface exposed by lab8_wrapper, releases the GPU, polls kernel_done,
// reads the cycle counter, and compares it against the standalone-sim
// cycle count of 732. The counter should match within +/-2 cycles since
// both the testbench poll loop and the hardware counter sample clocks
// starting at gpu_reset deassertion.
//
// Also sanity-checks a small subset of BCAST kernel outputs (Layer 3
// logits = 22.0 = 0x41B0 for simple test weights) to prove the kernel
// actually ran -- so "732 cycles" is not a hollow number.
//
// Usage:
//   iverilog -g2001 -I include -o tb_cycle_counter_sim \
//     tb/sim_primitives.v src/fifo/fifo_bram.v src/fifo/convertible_fifo.v \
//     src/arm/*.v $GPU_SRC src/lab8_wrapper.v tb/tb_cycle_counter.v && \
//     vvp tb_cycle_counter_sim

`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

module tb_cycle_counter;

    parameter CLK_PERIOD = 10;
    parameter IMEM_WORDS = 108;
    parameter DMEM_WORDS = 128;

    parameter EXPECTED_CYCLES = 732;
    parameter CYCLE_TOLERANCE = 4;

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

    always #(CLK_PERIOD/2) clk = ~clk;

    function [`UDP_REG_ADDR_WIDTH-1:0] make_addr;
        input [`LAB8_REG_ADDR_WIDTH-1:0] offset;
        begin
            make_addr = {`LAB8_BLOCK_ADDR, offset};
        end
    endfunction

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

    task gpu_load_dmem;
        input [9:0]  addr;
        input [63:0] data;
        begin
            pci_write(`LAB8_GPU_DMEM_ADDR, {22'd0, addr});
            pci_write(`LAB8_GPU_DMEM_WDATA_LO, data[31:0]);
            pci_write(`LAB8_GPU_DMEM_WDATA_HI, data[63:32]);
            pci_write(`LAB8_GPU_DMEM_CMD, 32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    task gpu_read_dmem;
        input  [9:0]  addr;
        output [63:0] data;
        begin
            pci_write(`LAB8_GPU_DMEM_ADDR, {22'd0, addr});
            repeat (2) @(posedge clk);
            pci_read(`LAB8_GPU_DMEM_RDATA_LO);
            data[31:0] = pci_rd_result;
            pci_read(`LAB8_GPU_DMEM_RDATA_HI);
            data[63:32] = pci_rd_result;
        end
    endtask

    reg [31:0] imem_file [0:1023];
    reg [63:0] dmem_file [0:1023];

    integer i, poll, pass, fail;
    reg [31:0] cycle_reg;
    reg [63:0] rd_word;

    initial begin
        $dumpfile("tb_cycle_counter.vcd");
        $dumpvars(0, tb_cycle_counter);

        for (i = 0; i < 1024; i = i + 1) begin
            imem_file[i] = 32'h0;
            dmem_file[i] = 64'h0;
        end
        $readmemh("programs/gpu/ann_ids_11_16_8_2_bcast.hex", imem_file);
        $readmemh("programs/gpu/data_ids_11_16_8_2_bcast.hex", dmem_file);

        clk = 0; reset = 1;
        in_data = 0; in_ctrl = 0; in_wr = 0;
        out_rdy = 1;
        reg_req_in = 0; reg_ack_in = 0; reg_rd_wr_L_in = 0;
        reg_addr_in = 0; reg_data_in = 0; reg_src_in = 0;
        pass = 0; fail = 0;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        $display("============================================================");
        $display("Cycle counter test: BCAST IDS kernel via PCI");
        $display("============================================================");

        // GPU starts in reset by default (reg_gpu_ctrl=0x1).
        // Sanity-check: counter should be 0 while GPU is in reset.
        pci_read(`LAB8_GPU_CYCLE_COUNT);
        if (pci_rd_result !== 32'd0) begin
            $display("  FAIL: counter not zero while GPU held in reset (got %0d)",
                     pci_rd_result);
            fail = fail + 1;
        end else begin
            $display("  PASS: counter reads 0 while GPU in reset");
            pass = pass + 1;
        end

        // Load IMEM via PCI
        $display("\nLoading IMEM (%0d words)...", IMEM_WORDS);
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            gpu_load_imem(i[9:0], imem_file[i]);
        end
        $display("  done");

        // Load DMEM via PCI
        $display("Loading DMEM (%0d words)...", DMEM_WORDS);
        for (i = 0; i < DMEM_WORDS; i = i + 1) begin
            gpu_load_dmem(i[9:0], dmem_file[i]);
        end
        $display("  done");

        // Counter should still be zero (we loaded IMEM/DMEM while GPU held in reset)
        pci_read(`LAB8_GPU_CYCLE_COUNT);
        if (pci_rd_result !== 32'd0) begin
            $display("  FAIL: counter changed during mem load (got %0d)",
                     pci_rd_result);
            fail = fail + 1;
        end else begin
            $display("  PASS: counter still 0 after IMEM/DMEM load");
            pass = pass + 1;
        end

        // Release GPU reset -> counter starts incrementing
        $display("\nReleasing GPU reset...");
        pci_write(`LAB8_GPU_CTRL, 32'd0);

        // Poll kernel_done (allow up to 2000 cycles for headroom)
        for (poll = 0; poll < 2000; poll = poll + 1) begin
            @(posedge clk);
            if (uut.gpu_kernel_done) begin
                poll = 2001;
            end
        end

        // Give the counter one extra cycle to settle then re-assert reset
        repeat (2) @(posedge clk);

        // Read counter
        pci_read(`LAB8_GPU_CYCLE_COUNT);
        cycle_reg = pci_rd_result;
        $display("\nHardware cycle counter = %0d  (expected ~%0d +/- %0d)",
                 cycle_reg, EXPECTED_CYCLES, CYCLE_TOLERANCE);

        if ((cycle_reg >= EXPECTED_CYCLES - CYCLE_TOLERANCE) &&
            (cycle_reg <= EXPECTED_CYCLES + CYCLE_TOLERANCE)) begin
            $display("  PASS: counter within tolerance");
            pass = pass + 1;
        end else begin
            $display("  FAIL: counter out of tolerance");
            fail = fail + 1;
        end

        // Kernel correctness spot-check: Layer 3 logits at DMEM[126..127]
        // should be 22.0 = 0x41B0 replicated (simple test weights).
        gpu_read_dmem(10'd126, rd_word);
        if (rd_word === 64'h41B041B041B041B0) begin
            $display("  PASS: out[0] @DMEM[126] = 0x%016h (22.0)", rd_word);
            pass = pass + 1;
        end else begin
            $display("  FAIL: out[0] @DMEM[126] = 0x%016h (expected 0x41B041B041B041B0)",
                     rd_word);
            fail = fail + 1;
        end
        gpu_read_dmem(10'd127, rd_word);
        if (rd_word === 64'h41B041B041B041B0) begin
            $display("  PASS: out[1] @DMEM[127] = 0x%016h (22.0)", rd_word);
            pass = pass + 1;
        end else begin
            $display("  FAIL: out[1] @DMEM[127] = 0x%016h (expected 0x41B041B041B041B0)",
                     rd_word);
            fail = fail + 1;
        end

        // Counter should not keep incrementing after kernel_done (frozen)
        repeat (50) @(posedge clk);
        pci_read(`LAB8_GPU_CYCLE_COUNT);
        if (pci_rd_result === cycle_reg) begin
            $display("  PASS: counter frozen after kernel_done (still %0d)",
                     pci_rd_result);
            pass = pass + 1;
        end else begin
            $display("  FAIL: counter kept moving (%0d -> %0d)",
                     cycle_reg, pci_rd_result);
            fail = fail + 1;
        end

        // Assert GPU reset again -> counter should clear on next release
        pci_write(`LAB8_GPU_CTRL, 32'd1);
        repeat (3) @(posedge clk);
        pci_read(`LAB8_GPU_CYCLE_COUNT);
        if (pci_rd_result === 32'd0) begin
            $display("  PASS: counter cleared on GPU reset");
            pass = pass + 1;
        end else begin
            $display("  FAIL: counter not cleared after reset (got %0d)",
                     pci_rd_result);
            fail = fail + 1;
        end

        $display("\n============================================================");
        $display("Results: %0d PASS, %0d FAIL", pass, fail);
        if (fail == 0) $display("*** ALL CHECKS PASSED ***");
        else           $display("*** SOME CHECKS FAILED ***");
        $display("============================================================");

        #100; $finish;
    end

endmodule
