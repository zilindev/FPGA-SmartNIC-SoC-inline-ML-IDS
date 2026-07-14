// tb_bcast.v -- Unit test for BCAST (cross-lane broadcast) instruction
//
// Loads bcast_test.hex kernel, data_bcast_test.hex data.
// Verifies all 4 lane selects produce correct broadcast results.
//
// Usage:
//   iverilog -g2001 -I include -o tb_bcast \
//     src/gpu/gpu_top.v src/gpu/core_top.v src/gpu/program_counter.v \
//     src/gpu/instruction_memory.v src/gpu/data_memory.v src/gpu/fetch_stage.v \
//     src/gpu/decode_stage.v src/gpu/instruction_decoder.v src/gpu/register_file.v \
//     src/gpu/pipeline_reg_idex.v src/gpu/pipeline_reg_ex.v src/gpu/pipeline_reg_exwb.v \
//     src/gpu/execute_stage.v src/gpu/simd_alu.v src/gpu/int16_alu.v \
//     src/gpu/bf16_alu.v src/gpu/bf16_fma_unit.v src/gpu/hazard_unit.v \
//     src/gpu/writeback_stage.v tb/tb_bcast.v && vvp tb_bcast

`timescale 1ns / 1ps

module tb_bcast;

    parameter CLK_PERIOD  = 8;
    parameter IMEM_WORDS  = 10;
    parameter DMEM_WORDS  = 5;
    parameter MAX_CYCLES  = 200;

    // Clock and reset
    reg clk, rst;
    always #(CLK_PERIOD/2) clk = ~clk;

    // GPU interface
    reg  [9:0]  ext_imem_addr;
    reg         ext_imem_wr_en;
    reg  [31:0] ext_imem_wr_data;
    wire [31:0] ext_imem_rd_data;

    reg  [9:0]  ext_dmem_addr;
    reg         ext_dmem_wr_en;
    reg  [63:0] ext_dmem_wr_data;
    wire [63:0] ext_dmem_rd_data;

    wire        kernel_done;

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

    // Hex file storage
    reg [31:0] imem_file [0:1023];
    reg [63:0] dmem_file [0:1023];

    // Test tracking
    integer pass_count, fail_count;
    integer cycle_count;
    integer i;
    reg [63:0] read_val;

    // Write one IMEM word
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

    // Write one DMEM word
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

    // Read one DMEM word
    task read_dmem;
        input  [9:0]  addr;
        output [63:0] data;
        begin
            @(posedge clk);
            ext_dmem_addr  = addr;
            ext_dmem_wr_en = 1'b0;
            @(posedge clk);
            data = ext_dmem_rd_data;
        end
    endtask

    // Check DMEM word against expected
    task check_dmem;
        input [9:0]  addr;
        input [63:0] expected;
        input [8*32-1:0] label;
        reg [63:0] actual;
        begin
            read_dmem(addr, actual);
            if (actual === expected) begin
                $display("  PASS  DMEM[%0d] = %016h  %0s", addr, actual, label);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL  DMEM[%0d] = %016h  expected %016h  %0s",
                         addr, actual, expected, label);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        // Init
        clk = 0;
        rst = 1;
        ext_imem_addr = 0;
        ext_imem_wr_en = 0;
        ext_imem_wr_data = 0;
        ext_dmem_addr = 0;
        ext_dmem_wr_en = 0;
        ext_dmem_wr_data = 0;
        pass_count = 0;
        fail_count = 0;

        // Load hex files
        for (i = 0; i < 1024; i = i + 1) begin
            imem_file[i] = 32'h00000000;
            dmem_file[i] = 64'h0000000000000000;
        end
        $readmemh("programs/gpu/bcast_test.hex", imem_file);
        $readmemh("programs/gpu/data_bcast_test.hex", dmem_file);

        // Hold reset during load
        repeat (5) @(posedge clk);

        $display("=== BCAST Instruction Unit Test ===");

        // Load IMEM (rst stays high -- core held in reset)
        $display("Loading IMEM (%0d words)...", IMEM_WORDS);
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            write_imem(i[9:0], imem_file[i]);
        end

        // Load DMEM
        $display("Loading DMEM (%0d words)...", DMEM_WORDS);
        for (i = 0; i < DMEM_WORDS; i = i + 1) begin
            write_dmem(i[9:0], dmem_file[i]);
        end

        // Verify loaded data
        read_dmem(10'd0, read_val);
        $display("  DMEM[0] = 0x%016h (expect DDDDCCCCBBBBAAAA)", read_val);

        // Release reset -- GPU starts executing from PC=0
        $display("Releasing GPU reset...");
        repeat (2) @(posedge clk);
        rst = 0;

        // Wait for kernel_done
        cycle_count = 0;
        while (!kernel_done && cycle_count < MAX_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        // Re-assert reset for safe DMEM reads
        rst = 1;
        repeat (2) @(posedge clk);

        if (cycle_count >= MAX_CYCLES) begin
            $display("TIMEOUT after %0d cycles", MAX_CYCLES);
            fail_count = fail_count + 1;
        end else begin
            $display("Kernel done in %0d cycles", cycle_count);
        end

        // Verify results
        $display("");
        $display("Input: DMEM[0] = DDDDCCCCBBBBAAAA");
        $display("  lane3=DDDD  lane2=CCCC  lane1=BBBB  lane0=AAAA");
        $display("");

        check_dmem(10'd1, 64'hAAAAAAAAAAAAAAAA, "BCAST lane 0");
        check_dmem(10'd2, 64'hBBBBBBBBBBBBBBBB, "BCAST lane 1");
        check_dmem(10'd3, 64'hCCCCCCCCCCCCCCCC, "BCAST lane 2");
        check_dmem(10'd4, 64'hDDDDDDDDDDDDDDDD, "BCAST lane 3");

        $display("");
        $display("====================================");
        $display("  %0d PASS, %0d FAIL (of 4 tests)", pass_count, fail_count);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");
        $display("====================================");

        $finish;
    end

endmodule
