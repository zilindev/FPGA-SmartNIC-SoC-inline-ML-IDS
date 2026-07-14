// tb_ids_inference.v -- Standalone GPU testbench for IDS MLP (11->16->8->2)
//
// Loads kernel from ann_ids_11_16_8_2.hex, data from data_ids_11_16_8_2.hex.
// Verifies all layer outputs against known expected values.
//
// Usage:
//   iverilog -g2001 -I include -o tb_ids \
//     src/gpu/gpu_top.v src/gpu/core_top.v src/gpu/program_counter.v \
//     src/gpu/instruction_memory.v src/gpu/data_memory.v src/gpu/fetch_stage.v \
//     src/gpu/decode_stage.v src/gpu/instruction_decoder.v src/gpu/register_file.v \
//     src/gpu/pipeline_reg_idex.v src/gpu/pipeline_reg_ex.v src/gpu/pipeline_reg_exwb.v \
//     src/gpu/execute_stage.v src/gpu/simd_alu.v src/gpu/int16_alu.v \
//     src/gpu/bf16_alu.v src/gpu/bf16_fma_unit.v src/gpu/hazard_unit.v \
//     src/gpu/writeback_stage.v tb/tb_ids_inference.v && vvp tb_ids

`timescale 1ns / 1ps

module tb_ids_inference;

    // ----------------------------------------------------------------
    // Parameters
    // ----------------------------------------------------------------
    parameter CLK_PERIOD  = 8;    // 125 MHz
    parameter IMEM_WORDS  = 211;  // kernel size
    parameter DMEM_WORDS  = 383;  // data size
    parameter MAX_CYCLES  = 10000;

    // DMEM addresses for verification
    parameter H1_BASE  = 203;   // Layer 1 output [203..218]
    parameter H2_BASE  = 355;   // Layer 2 output [355..362]
    parameter OUT_BASE = 381;   // Layer 3 output [381..382]

    // ----------------------------------------------------------------
    // Clock and reset
    // ----------------------------------------------------------------
    reg clk, rst;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ----------------------------------------------------------------
    // GPU external interface signals
    // ----------------------------------------------------------------
    reg  [9:0]  ext_imem_addr;
    reg         ext_imem_wr_en;
    reg  [31:0] ext_imem_wr_data;
    wire [31:0] ext_imem_rd_data;

    reg  [9:0]  ext_dmem_addr;
    reg         ext_dmem_wr_en;
    reg  [63:0] ext_dmem_wr_data;
    wire [63:0] ext_dmem_rd_data;

    wire        kernel_done;

    // ----------------------------------------------------------------
    // GPU instantiation
    // ----------------------------------------------------------------
    gpu_top uut (
        .clk             (clk),
        .rst             (rst),
        .thread_id       (32'd0),
        .block_id        (32'd0),
        .block_dim       (32'd1),
        .kernel_done     (kernel_done),
        .ext_imem_addr   (ext_imem_addr),
        .ext_imem_wr_en  (ext_imem_wr_en),
        .ext_imem_wr_data(ext_imem_wr_data),
        .ext_imem_rd_data(ext_imem_rd_data),
        .ext_dmem_addr   (ext_dmem_addr),
        .ext_dmem_wr_en  (ext_dmem_wr_en),
        .ext_dmem_wr_data(ext_dmem_wr_data),
        .ext_dmem_rd_data(ext_dmem_rd_data)
    );

    // ----------------------------------------------------------------
    // Hex file storage
    // ----------------------------------------------------------------
    reg [31:0] imem_file [0:1023];
    reg [63:0] dmem_file [0:1023];

    // ----------------------------------------------------------------
    // Test state
    // ----------------------------------------------------------------
    integer i;
    integer cycle_count;
    integer pass_count, fail_count;
    reg [63:0] read_val;

    // ----------------------------------------------------------------
    // Tasks
    // ----------------------------------------------------------------

    // Write one word to GPU IMEM via external port
    task write_imem;
        input [9:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            ext_imem_addr    = addr;
            ext_imem_wr_data = data;
            ext_imem_wr_en   = 1'b1;
            @(posedge clk);
            ext_imem_wr_en   = 1'b0;
        end
    endtask

    // Write one word to GPU DMEM via external port
    task write_dmem;
        input [9:0]  addr;
        input [63:0] data;
        begin
            @(posedge clk);
            ext_dmem_addr    = addr;
            ext_dmem_wr_data = data;
            ext_dmem_wr_en   = 1'b1;
            @(posedge clk);
            ext_dmem_wr_en   = 1'b0;
        end
    endtask

    // Read one word from GPU DMEM via external port (synchronous BRAM)
    task read_dmem;
        input  [9:0]  addr;
        output [63:0] data;
        begin
            @(posedge clk);
            ext_dmem_addr = addr;
            ext_dmem_wr_en = 1'b0;
            @(posedge clk);  // 1-cycle BRAM latency
            data = ext_dmem_rd_data;
        end
    endtask

    // Check a DMEM word against expected, print result
    task check_dmem;
        input [9:0]  addr;
        input [63:0] expected;
        input [8*32-1:0] label;  // string label (up to 32 chars)
        reg [63:0] actual;
        begin
            read_dmem(addr, actual);
            if (actual === expected) begin
                $display("  PASS  DMEM[%0d] %s = 0x%016h", addr, label, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL  DMEM[%0d] %s = 0x%016h (expected 0x%016h)",
                         addr, label, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ----------------------------------------------------------------
    // Main test sequence
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("tb_ids_inference.vcd");
        $dumpvars(0, tb_ids_inference);

        // Initialize hex file arrays to zero
        for (i = 0; i < 1024; i = i + 1) begin
            imem_file[i] = 32'h0;
            dmem_file[i] = 64'h0;
        end

        // Load hex files
        $readmemh("programs/gpu/ann_ids_11_16_8_2.hex", imem_file);
        $readmemh("programs/gpu/data_ids_11_16_8_2.hex", dmem_file);

        // ============================================================
        // Reset
        // ============================================================
        clk = 0;
        rst = 1;
        ext_imem_wr_en   = 0;
        ext_dmem_wr_en   = 0;
        ext_imem_addr    = 0;
        ext_dmem_addr    = 0;
        ext_imem_wr_data = 0;
        ext_dmem_wr_data = 0;
        pass_count = 0;
        fail_count = 0;

        repeat (5) @(posedge clk);

        $display("============================================================");
        $display("IDS MLP Inference Test (11 -> 16 -> 8 -> 2)");
        $display("  Kernel: ann_ids_11_16_8_2.hex (%0d instructions)", IMEM_WORDS);
        $display("  Data:   data_ids_11_16_8_2.hex (%0d words)", DMEM_WORDS);
        $display("============================================================");

        // ============================================================
        // Load IMEM (kernel)
        // ============================================================
        $display("\nLoading IMEM...");
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            write_imem(i[9:0], imem_file[i]);
        end
        $display("  Loaded %0d instructions", IMEM_WORDS);

        // ============================================================
        // Load DMEM (weights, biases, inputs)
        // ============================================================
        $display("Loading DMEM...");
        for (i = 0; i < DMEM_WORDS; i = i + 1) begin
            write_dmem(i[9:0], dmem_file[i]);
        end
        $display("  Loaded %0d data words", DMEM_WORDS);

        // ============================================================
        // Verify a few loaded values
        // ============================================================
        $display("\nVerifying loaded data...");
        // Check input x[0] at DMEM[0]
        read_dmem(10'd0, read_val);
        $display("  DMEM[0] (x[0]) = 0x%016h", read_val);
        // Check first weight w1[0][0] at DMEM[11]
        read_dmem(10'd11, read_val);
        $display("  DMEM[11] (w1[0][0]) = 0x%016h", read_val);

        // ============================================================
        // Release reset -- GPU starts executing
        // ============================================================
        $display("\nReleasing GPU reset...");
        repeat (2) @(posedge clk);
        rst = 0;

        // ============================================================
        // Wait for kernel_done
        // ============================================================
        cycle_count = 0;
        while (!kernel_done && cycle_count < MAX_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        // Re-assert reset to stop GPU (safe for DMEM reads)
        rst = 1;
        repeat (2) @(posedge clk);

        if (cycle_count >= MAX_CYCLES) begin
            $display("\n*** TIMEOUT: GPU did not halt within %0d cycles ***", MAX_CYCLES);
            fail_count = fail_count + 1;
        end else begin
            $display("\nGPU halted after %0d cycles (~%0d ns)", cycle_count,
                     cycle_count * CLK_PERIOD);
        end

        // ============================================================
        // Verify Layer 1 outputs (h1[0..15] at DMEM[203..218])
        // Simple test: all weights=0.25, biases=0, inputs=1.0
        // Expected: ReLU(11 * 0.25 * 1.0) = 2.75 -> BF16 0x4030
        // ============================================================
        $display("\n--- Layer 1 outputs (h1[0..15], expected 2.75 = 0x4030) ---");
        for (i = 0; i < 16; i = i + 1) begin
            check_dmem(H1_BASE + i, 64'h4030403040304030, "h1");
        end

        // ============================================================
        // Verify Layer 2 outputs (h2[0..7] at DMEM[355..362])
        // Expected: ReLU(16 * 0.25 * 2.75) = 11.0 -> BF16 0x4130
        // ============================================================
        $display("\n--- Layer 2 outputs (h2[0..7], expected 11.0 = 0x4130) ---");
        for (i = 0; i < 8; i = i + 1) begin
            check_dmem(H2_BASE + i, 64'h4130413041304130, "h2");
        end

        // ============================================================
        // Verify Layer 3 outputs (out[0..1] at DMEM[381..382])
        // Expected: 8 * 0.25 * 11.0 = 22.0 -> BF16 0x41B0
        // ============================================================
        $display("\n--- Layer 3 outputs (out[0..1], expected 22.0 = 0x41B0) ---");
        check_dmem(OUT_BASE,     64'h41B041B041B041B0, "out[0]");
        check_dmem(OUT_BASE + 1, 64'h41B041B041B041B0, "out[1]");

        // ============================================================
        // Summary
        // ============================================================
        $display("\n============================================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        if (fail_count == 0)
            $display("*** ALL TESTS PASSED ***");
        else
            $display("*** SOME TESTS FAILED ***");
        $display("============================================================");

        #100;
        $finish;
    end

endmodule
