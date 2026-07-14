`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_arm_reg_access -- Milestone 4 §9.5.1 unit test for the register-alias
// window introduced in milestone3_refinements.md §9.3. Confirms the new
// 0xF0..0xFF window in src/arm/data_mem.v reaches every wrapper register
// it claims to.
//
// Test plan:
//   1. ARM program writes a unique pattern to each of the 7 RW aliases
//      (DMA_CTRL/FIFO_ADDR/GPU_ADDR/LEN, GPU_CTRL, FIFO_MODE, CTRL).
//   2. ARM then reads each of the 3 RO aliases (GPU_STATUS, FIFO_STATUS,
//      CYCLE_COUNT) into DMEM[0..2].
//   3. ARM HALTs.
//   4. PCI reads each RW register and asserts pattern coherence.
//   5. PCI reads DMEM[0..2] and asserts the RO values match the wrapper
//      state we set up (GPU held in reset, FIFO empty, counter = 0).
//
// All 4 threads run the same program (FGMT). Since each thread writes the
// same pattern and reads the same deterministic state, results converge.
//
// Patterns are chosen so trigger bits stay clear:
//   - DMA_CTRL bit[0]=0 (skips the auto-clear strobe)
//   - GPU_CTRL bit[0]=1 (keeps GPU in reset so cycle counter stays 0)
//   - CTRL    bit[0]=0 (skips soft-reset of the FIFO)
//   - FIFO_MODE = 2 (DMA mode, but DMA never starts)

module tb_arm_reg_access;

    reg         clk, reset;

    // NF2.1 data path (unused but required by lab8_wrapper)
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

    // ---------------------------------------------------------------
    // PCI helpers
    // ---------------------------------------------------------------
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

    task load_imem;
        input [8:0]  addr;
        input [31:0] data;
        begin
            pci_write(`LAB8_CPU_IMEM_ADDR,  {23'd0, addr});
            pci_write(`LAB8_CPU_IMEM_WDATA, data);
            pci_write(`LAB8_CPU_IMEM_CMD,   32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------
    // ARM program
    //   r0 = 0xF0  (alias-window base pointer)
    //   r3 = 0x00  (DMEM base pointer)
    //   r1 used as the immediate value being stored to each alias
    //   r2 used to ferry RO alias reads into DMEM[0..2]
    // ---------------------------------------------------------------
    localparam NUM_INST = 23;
    reg [31:0] program_mem [0:NUM_INST-1];

    // Patterns (all fit in MOV imm8 with rotate=0)
    localparam [31:0] PAT_DMA_CTRL  = 32'h00000002; // bit[0]=0 -> no auto-clear
    localparam [31:0] PAT_DMA_FIFO  = 32'h00000042;
    localparam [31:0] PAT_DMA_GPU   = 32'h00000053;
    localparam [31:0] PAT_DMA_LEN   = 32'h00000064;
    localparam [31:0] PAT_GPU_CTRL  = 32'h00000081; // bit[0]=1 -> GPU stays held
    localparam [31:0] PAT_FIFO_MODE = 32'h00000002;
    localparam [31:0] PAT_CTRL      = 32'h00000080; // bit[0]=0 -> no soft-reset

    // Expected RO values (with GPU held in reset, FIFO empty, no DMA)
    //   GPU_STATUS  = {30'd0, dma_active=0, kernel_done=0}                = 0x00
    //   FIFO_STATUS = {pkt_word_cnt=0, head=0, tail=0, 5'd0, empty=1,
    //                  full=0, pkt_ready=0}                               = 0x04
    //   CYCLE_COUNT = held at 0 because gpu_reset=1 throughout            = 0x00
    localparam [31:0] EXP_GPU_STATUS  = 32'h00000000;
    localparam [31:0] EXP_FIFO_STATUS = 32'h00000004;
    localparam [31:0] EXP_CYCLE_COUNT = 32'h00000000;

    integer i, j, pass;

    initial begin
        $dumpfile("tb_arm_reg_access.vcd");
        $dumpvars(0, tb_arm_reg_access);

        // ARM instruction list (assembler done by hand — mnemonics on the right)
        program_mem[0]  = 32'hE3A000F0; // MOV r0, #0xF0
        program_mem[1]  = 32'hE3A03000; // MOV r3, #0x00
        program_mem[2]  = 32'hE3A01002; // MOV r1, #0x02
        program_mem[3]  = 32'hE5801000; // STR r1, [r0, #0x00]   ; DMA_CTRL
        program_mem[4]  = 32'hE3A01042; // MOV r1, #0x42
        program_mem[5]  = 32'hE5801001; // STR r1, [r0, #0x01]   ; DMA_FIFO_ADDR
        program_mem[6]  = 32'hE3A01053; // MOV r1, #0x53
        program_mem[7]  = 32'hE5801002; // STR r1, [r0, #0x02]   ; DMA_GPU_ADDR
        program_mem[8]  = 32'hE3A01064; // MOV r1, #0x64
        program_mem[9]  = 32'hE5801003; // STR r1, [r0, #0x03]   ; DMA_LENGTH
        program_mem[10] = 32'hE3A01081; // MOV r1, #0x81
        program_mem[11] = 32'hE5801004; // STR r1, [r0, #0x04]   ; GPU_CTRL
        program_mem[12] = 32'hE3A01002; // MOV r1, #0x02
        program_mem[13] = 32'hE5801007; // STR r1, [r0, #0x07]   ; FIFO_MODE
        program_mem[14] = 32'hE3A01080; // MOV r1, #0x80
        program_mem[15] = 32'hE5801008; // STR r1, [r0, #0x08]   ; CTRL
        program_mem[16] = 32'hE5902005; // LDR r2, [r0, #0x05]   ; GPU_STATUS
        program_mem[17] = 32'hE5832000; // STR r2, [r3, #0x00]   ; -> DMEM[0]
        program_mem[18] = 32'hE5902006; // LDR r2, [r0, #0x06]   ; FIFO_STATUS
        program_mem[19] = 32'hE5832001; // STR r2, [r3, #0x01]   ; -> DMEM[1]
        program_mem[20] = 32'hE5902009; // LDR r2, [r0, #0x09]   ; CYCLE_COUNT
        program_mem[21] = 32'hE5832002; // STR r2, [r3, #0x02]   ; -> DMEM[2]
        program_mem[22] = 32'hFFFFFFFF; // HALT

        // Init
        clk            = 0;
        reset          = 1;
        in_data        = 64'd0;
        in_ctrl        = 8'd0;
        in_wr          = 1'b0;
        out_rdy        = 1'b1;
        reg_req_in     = 1'b0;
        reg_ack_in     = 1'b0;
        reg_rd_wr_L_in = 1'b0;
        reg_addr_in    = {`UDP_REG_ADDR_WIDTH{1'b0}};
        reg_data_in    = 32'd0;
        reg_src_in     = 2'd0;
        pass           = 1;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        // -----------------------------------------------------------
        $display("\n=== Step 1: hold CPU in reset, load IMEM for all 4 threads ===");

        pci_write(`LAB8_CPU_CTRL, 32'h2);           // bit[1]=1 asserts cpu_reset
        repeat (3) @(posedge clk);

        // Each FGMT thread fetches from its own 128-word IMEM region.
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < NUM_INST; j = j + 1) begin
                load_imem(i * 128 + j, program_mem[j]);
            end
        end

        // Spot-check IMEM[0]
        pci_write(`LAB8_CPU_IMEM_ADDR, 32'd0);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_CPU_IMEM_RDATA);
        if (pci_rd_result !== program_mem[0]) begin
            $display("FAIL: IMEM[0] readback = 0x%08h, expected 0x%08h",
                     pci_rd_result, program_mem[0]);
            pass = 0;
        end else
            $display("PASS: IMEM[0] readback = 0x%08h", pci_rd_result);

        // -----------------------------------------------------------
        $display("\n=== Step 2: release CPU, wait for all_halted ===");

        pci_write(`LAB8_CPU_CTRL, 32'h1);           // bit[1]=0 release, bit[0]=1 run

        // 23 inst x 4 threads x 4 cycles/thread + drain ~= 100 cycles. 500 = generous.
        for (i = 0; i < 500; i = i + 1) begin
            @(posedge clk);
            if (uut.cpu_all_halted) begin
                $display("CPU halted after %0d cycles", i);
                i = 500;
            end
        end

        if (!uut.cpu_all_halted) begin
            $display("FAIL: CPU did not halt within 500 cycles");
            pass = 0;
        end

        pci_write(`LAB8_CPU_CTRL, 32'h0);           // stop CPU
        repeat (2) @(posedge clk);

        // -----------------------------------------------------------
        $display("\n=== Step 3: PCI verifies the 7 RW alias writes ===");

        pci_read(`LAB8_DMA_CTRL);
        if (pci_rd_result !== PAT_DMA_CTRL) begin
            $display("FAIL: DMA_CTRL = 0x%08h, expected 0x%08h", pci_rd_result, PAT_DMA_CTRL);
            pass = 0;
        end else
            $display("PASS: DMA_CTRL = 0x%08h", pci_rd_result);

        pci_read(`LAB8_DMA_FIFO_ADDR);
        if (pci_rd_result !== PAT_DMA_FIFO) begin
            $display("FAIL: DMA_FIFO_ADDR = 0x%08h, expected 0x%08h", pci_rd_result, PAT_DMA_FIFO);
            pass = 0;
        end else
            $display("PASS: DMA_FIFO_ADDR = 0x%08h", pci_rd_result);

        pci_read(`LAB8_DMA_GPU_ADDR);
        if (pci_rd_result !== PAT_DMA_GPU) begin
            $display("FAIL: DMA_GPU_ADDR = 0x%08h, expected 0x%08h", pci_rd_result, PAT_DMA_GPU);
            pass = 0;
        end else
            $display("PASS: DMA_GPU_ADDR = 0x%08h", pci_rd_result);

        pci_read(`LAB8_DMA_LENGTH);
        if (pci_rd_result !== PAT_DMA_LEN) begin
            $display("FAIL: DMA_LENGTH = 0x%08h, expected 0x%08h", pci_rd_result, PAT_DMA_LEN);
            pass = 0;
        end else
            $display("PASS: DMA_LENGTH = 0x%08h", pci_rd_result);

        pci_read(`LAB8_GPU_CTRL);
        if (pci_rd_result !== PAT_GPU_CTRL) begin
            $display("FAIL: GPU_CTRL = 0x%08h, expected 0x%08h", pci_rd_result, PAT_GPU_CTRL);
            pass = 0;
        end else
            $display("PASS: GPU_CTRL = 0x%08h", pci_rd_result);

        pci_read(`LAB8_FIFO_MODE);
        if (pci_rd_result !== PAT_FIFO_MODE) begin
            $display("FAIL: FIFO_MODE = 0x%08h, expected 0x%08h", pci_rd_result, PAT_FIFO_MODE);
            pass = 0;
        end else
            $display("PASS: FIFO_MODE = 0x%08h", pci_rd_result);

        pci_read(`LAB8_CTRL);
        if (pci_rd_result !== PAT_CTRL) begin
            $display("FAIL: CTRL = 0x%08h, expected 0x%08h", pci_rd_result, PAT_CTRL);
            pass = 0;
        end else
            $display("PASS: CTRL = 0x%08h", pci_rd_result);

        // -----------------------------------------------------------
        $display("\n=== Step 4: PCI verifies the 3 RO alias reads via DMEM ===");

        // DMEM[0] should hold GPU_STATUS that the ARM read.
        pci_write(`LAB8_CPU_DMEM_ADDR, 32'd0);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_CPU_DMEM_RDATA_LO);
        if (pci_rd_result !== EXP_GPU_STATUS) begin
            $display("FAIL: DMEM[0] (GPU_STATUS) = 0x%08h, expected 0x%08h",
                     pci_rd_result, EXP_GPU_STATUS);
            pass = 0;
        end else
            $display("PASS: DMEM[0] (GPU_STATUS) = 0x%08h", pci_rd_result);

        // DMEM[1] should hold FIFO_STATUS.
        pci_write(`LAB8_CPU_DMEM_ADDR, 32'd1);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_CPU_DMEM_RDATA_LO);
        if (pci_rd_result !== EXP_FIFO_STATUS) begin
            $display("FAIL: DMEM[1] (FIFO_STATUS) = 0x%08h, expected 0x%08h",
                     pci_rd_result, EXP_FIFO_STATUS);
            pass = 0;
        end else
            $display("PASS: DMEM[1] (FIFO_STATUS) = 0x%08h", pci_rd_result);

        // DMEM[2] should hold CYCLE_COUNT.
        pci_write(`LAB8_CPU_DMEM_ADDR, 32'd2);
        repeat (2) @(posedge clk);
        pci_read(`LAB8_CPU_DMEM_RDATA_LO);
        if (pci_rd_result !== EXP_CYCLE_COUNT) begin
            $display("FAIL: DMEM[2] (CYCLE_COUNT) = 0x%08h, expected 0x%08h",
                     pci_rd_result, EXP_CYCLE_COUNT);
            pass = 0;
        end else
            $display("PASS: DMEM[2] (CYCLE_COUNT) = 0x%08h", pci_rd_result);

        // -----------------------------------------------------------
        $display("\n=== Result ===");
        if (pass) $display("ALL TESTS PASSED  (10/10 alias slots coherent)");
        else      $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
