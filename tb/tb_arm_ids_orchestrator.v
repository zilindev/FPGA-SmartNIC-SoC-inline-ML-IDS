`timescale 1ns / 1ps
`include "lab8_reg_defines.vh"

// tb_arm_ids_orchestrator -- ARM orchestrator with Fix B on-chip egress.
//
// Extends the Fix A baseline (§13.2) to close the round-trip: the orchestrator
// now rewrites Ethernet / IP / UDP headers in BRAM[1..6], fixes the NF2.1
// header at BRAM[0] so the response egresses via MAC0, and triggers drain
// through the new reg_fifo_drain alias slot (0xFA). The TB captures the
// NF2 output stream and asserts word-by-word.
//
// Stack:
//   - GPU IMEM <- BCAST IDS kernel (ann_ids_11_16_8_2_bcast.hex, 108 inst)
//   - GPU DMEM <- IDS test data    (data_ids_11_16_8_2_bcast.hex, 128 words)
//   - ARM IMEM <- ids_orchestrator.hex (117 inst on T0; T1-T3 just HALT)
//   - FIFO     <- one IDS feature packet (18 NF2 words, realistic headers)
//
// Per iteration the orchestrator:
//   1.  wait_pkt:     poll FIFO_STATUS until pkt_ready
//   2.               FIFO_MODE=2; DMA FIFO[7..17] -> GPU DMEM[0..10]
//   3.  wait_dma_in: poll GPU_STATUS until !dma_busy
//   4.               GPU_CTRL=0 (release)
//   5.  wait_kernel: poll GPU_STATUS until kernel_done
//   6.               GPU_CTRL=1 (hold); DMA GPU DMEM[126..127] -> FIFO[7..8]
//   7.  wait_dma_out: poll GPU_STATUS until !dma_busy
//   8.               FIFO_MODE=1; MAC swap BRAM[1..2]; IP/UDP swap BRAM[4..5];
//                    zero UDP cksum BRAM[6]; NF2 hdr fixup BRAM[0]
//   9.               FIFO_MODE=0; FIFO_DRAIN=1 (alias slot 0xFA)
//  10.  wait_drain: poll FIFO_STATUS until pkt_ready clears
//  11.  branch back to wait_pkt
//
// Test 1: single-packet end-to-end egress.
// Test 2: 3-packet stress (auto-clear, mode flips, back-to-back drains).

module tb_arm_ids_orchestrator;

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

    // ---------------------------------------------------------------
    // NF2 egress capture (passive listener)
    // Keeps a 32-deep history of each drained frame so Test 1 and Test 2
    // can examine the post-header-rewrite stream word-by-word.
    // ---------------------------------------------------------------
    parameter CAP_DEPTH = 32;
    reg [71:0] cap_buf [0:CAP_DEPTH-1];
    integer    cap_cnt;
    integer    cap_reset_at;

    always @(posedge clk) begin
        if (reset) begin
            cap_cnt <= 0;
        end else if (out_wr && (cap_cnt < CAP_DEPTH)) begin
            cap_buf[cap_cnt] <= {out_ctrl, out_data};
            cap_cnt          <= cap_cnt + 1;
        end
    end

    task cap_clear;
        integer i;
        begin
            for (i = 0; i < CAP_DEPTH; i = i + 1) cap_buf[i] = 72'd0;
            cap_cnt = 0;
        end
    endtask

    // ---------------------------------------------------------------
    // PCI / FIFO helpers
    // ---------------------------------------------------------------
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

    task gpu_load_imem;
        input [9:0]  addr;
        input [31:0] data;
        begin
            pci_write(`LAB8_GPU_IMEM_ADDR, {22'd0, addr});
            pci_write(`LAB8_GPU_IMEM_WDATA, data);
            pci_write(`LAB8_GPU_IMEM_CMD,   32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    task gpu_load_dmem;
        input [9:0]  addr;
        input [63:0] data;
        begin
            pci_write(`LAB8_GPU_DMEM_ADDR,     {22'd0, addr});
            pci_write(`LAB8_GPU_DMEM_WDATA_LO, data[31:0]);
            pci_write(`LAB8_GPU_DMEM_WDATA_HI, data[63:32]);
            pci_write(`LAB8_GPU_DMEM_CMD,      32'h1);
            repeat (2) @(posedge clk);
        end
    endtask

    task arm_load_imem;
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
    // Files + arrays
    // ---------------------------------------------------------------
    parameter IMEM_WORDS = 108;
    parameter DMEM_WORDS = 128;
    parameter NUM_ORCH   = 385;   // full .hex length (T0 program + HALT padding through idx 384)

    reg [31:0] kernel_imem_file [0:1023];
    reg [63:0] kernel_dmem_file [0:1023];
    reg [31:0] orch_file        [0:NUM_ORCH-1];

    initial $readmemh("programs/gpu/ann_ids_11_16_8_2_bcast.hex", kernel_imem_file);
    initial $readmemh("programs/gpu/data_ids_11_16_8_2_bcast.hex", kernel_dmem_file);
    initial $readmemh("programs/arm/ids_orchestrator.hex",        orch_file);

    // ---------------------------------------------------------------
    // Crafted ingress packet with realistic Ethernet / IP / UDP headers.
    // Layout (byte positions in each 64-bit word, MSB first):
    //   BRAM[0] NF2.1 hdr : dst_port=0x0002 word_len=0x0012 src_port=0x0001 byte_len=0x0090
    //   BRAM[1]           : DstMAC[0..5]=FF:FF:FF:FF:FF:FF, SrcMAC[0..1]=AA:BB
    //   BRAM[2]           : SrcMAC[2..5]=CC:DD:EE:FF, EtherType=0x0800, IPverIHL=0x45, TOS=0x00
    //   BRAM[3]           : IP total_len=0x0082, ID=0xABCD, flags/frag=0x0000, TTL=0x40, protocol=0x11(UDP)
    //   BRAM[4]           : IP cksum=0x1234, SrcIP=10.0.4.2, DstIP[0..1]=0x0A,0x00
    //   BRAM[5]           : DstIP[2..3]=0x07,0xFF, UDPsrc=0x1111, UDPdst=0x270F, UDPlen=0x006E
    //   BRAM[6]           : UDPcksum=0x5678, payload[0..5]=0x99,0x88,0x77,0x66,0x55,0x44
    //   BRAM[7..17]       : 11 replicated BF16 feature words (all 1.0 -> kernel output 22.0)
    //
    // Expected after Fix B rewrites:
    //   BRAM[0] NF2.1 hdr : dst_port=0x0001 (MAC0), src_port=0x0002 (CPU0), lens preserved; ctrl=0xFF
    //   BRAM[1]           : DstMAC=AA:BB:CC:DD:EE:FF, SrcMAC[0..1]=FF:FF
    //   BRAM[2]           : SrcMAC[2..5]=FF:FF:FF:FF, EtherType..TOS unchanged
    //   BRAM[4]           : IP cksum unchanged, SrcIP=10.0.7.255, DstIP[0..1]=0x0A,0x00
    //   BRAM[5]           : DstIP[2..3]=0x04,0x02, UDPsrc=0x270F, UDPdst=0x1111, UDPlen unchanged
    //   BRAM[6]           : UDPcksum=0x0000, payload unchanged (0)
    //   BRAM[7..8]        : 4x-replicated BF16 logits = 0x41B041B041B041B0 each
    //   BRAM[9..17]       : unchanged (original feature words, now garbage in response)
    // ---------------------------------------------------------------
    localparam [63:0] PKT_HEADER  = 64'h0002_0012_0001_0090;
    localparam [63:0] PKT_ETH1    = 64'hFFFF_FFFF_FFFF_AABB;
    localparam [63:0] PKT_ETH2    = 64'hCCDD_EEFF_0800_4500;
    localparam [63:0] PKT_IPHDR1  = 64'h0082_ABCD_0000_4011;
    localparam [63:0] PKT_IPHDR2  = 64'h1234_0A00_0402_0A00;
    localparam [63:0] PKT_UDPHDR  = 64'h07FF_1111_270F_006E;
    localparam [63:0] PKT_UDP_CK  = 64'h5678_9988_7766_5544;
    localparam [63:0] FEAT_1P0    = 64'h3F80_3F80_3F80_3F80; // 4xBF16(1.0)

    // Post-swap expected values (pre-drain, post-header-rewrite):
    localparam [63:0] EXP_HDR     = 64'h0001_0012_0002_0090;
    localparam [63:0] EXP_ETH1    = 64'hAABB_CCDD_EEFF_FFFF;
    localparam [63:0] EXP_ETH2    = 64'hFFFF_FFFF_0800_4500;
    localparam [63:0] EXP_IPHDR2  = 64'h1234_0A00_07FF_0A00;
    localparam [63:0] EXP_UDPHDR  = 64'h0402_270F_1111_006E;
    localparam [63:0] EXP_UDP_CK  = 64'h0000_0000_0000_0000;
    localparam [63:0] EXP_LOGIT   = 64'h41B0_41B0_41B0_41B0; // 4xBF16(22.0)

    task inject_packet_1p0;
        begin
            fifo_write(8'hFF, PKT_HEADER);
            fifo_write(8'h00, PKT_ETH1);    // BRAM[1]
            fifo_write(8'h00, PKT_ETH2);    // BRAM[2]
            fifo_write(8'h00, PKT_IPHDR1);  // BRAM[3]
            fifo_write(8'h00, PKT_IPHDR2);  // BRAM[4]
            fifo_write(8'h00, PKT_UDPHDR);  // BRAM[5]
            fifo_write(8'h00, PKT_UDP_CK);  // BRAM[6]
            // BRAM[7..17]: 11 replicated features at 1.0
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
            fifo_write(8'h00, FEAT_1P0);
        end
    endtask

    // ---------------------------------------------------------------
    // Per-word assertion helper
    // ---------------------------------------------------------------
    task check_word;
        input integer    idx;
        input [7:0]      exp_ctrl;
        input [63:0]     exp_data;
        input [255*8:1]  label;
        reg   [71:0]     got;
        begin
            got = cap_buf[idx];
            if (got === {exp_ctrl, exp_data})
                $display("    PASS: word[%0d] (%0s) ctrl=0x%02h data=0x%016h",
                         idx, label, got[71:64], got[63:0]);
            else begin
                $display("    FAIL: word[%0d] (%0s) got ctrl=0x%02h data=0x%016h; expected ctrl=0x%02h data=0x%016h",
                         idx, label, got[71:64], got[63:0], exp_ctrl, exp_data);
                pass = 0;
            end
        end
    endtask

    // ---------------------------------------------------------------
    // Test bookkeeping
    // ---------------------------------------------------------------
    integer i, k;
    integer pass;

    initial begin
        $dumpfile("tb_arm_ids_orchestrator.vcd");
        $dumpvars(0, tb_arm_ids_orchestrator);

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

        for (i = IMEM_WORDS; i < 1024; i = i + 1) kernel_imem_file[i] = 32'd0;
        for (i = DMEM_WORDS; i < 1024; i = i + 1) kernel_dmem_file[i] = 64'd0;
        cap_clear;

        repeat (5) @(posedge clk);
        reset = 0;
        repeat (2) @(posedge clk);

        $display("\n=== Stage 1: hold ARM in reset (CPU_CTRL=2) ===");
        pci_write(`LAB8_CPU_CTRL, 32'h2);
        repeat (3) @(posedge clk);

        $display("\n=== Stage 2: load BCAST IDS kernel into GPU IMEM ===");
        for (i = 0; i < IMEM_WORDS; i = i + 1) begin
            gpu_load_imem(i[9:0], kernel_imem_file[i]);
        end
        $display("  loaded %0d GPU IMEM words", IMEM_WORDS);

        $display("\n=== Stage 3: load IDS test data into GPU DMEM ===");
        for (i = 0; i < DMEM_WORDS; i = i + 1) begin
            gpu_load_dmem(i[9:0], kernel_dmem_file[i]);
        end
        $display("  loaded %0d GPU DMEM words", DMEM_WORDS);

        $display("\n=== Stage 4: load full orchestrator into ARM IMEM from hex ===");
        for (i = 0; i < NUM_ORCH; i = i + 1) begin
            arm_load_imem(i[8:0], orch_file[i]);
        end
        $display("  loaded %0d ARM IMEM words (T0 orch + HALT padding)", NUM_ORCH);

        $display("\n=== Stage 5: FIFO_MODE=0, release ARM (CPU_CTRL=1) ===");
        pci_write(`LAB8_FIFO_MODE, 32'h0);
        pci_write(`LAB8_CPU_CTRL,  32'h1);
        repeat (20) @(posedge clk);   // let T0 finish setup + settle into wait_pkt

        // ===========================================================
        $display("\n###############################################");
        $display("# Test 1: single-packet end-to-end egress");
        $display("###############################################");

        cap_clear;
        $display("\n  Injecting packet 1 (18 words, features = 1.0)...");
        inject_packet_1p0;

        // Wait flat budget: setup + DMA in + kernel (~750c) + DMA out +
        // ~120 ARM cycles of header munge (at 4 cyc/instr FGMT) + drain.
        // 8000 cycles covers comfortably.
        repeat (8000) @(posedge clk);

        $display("\n  Captured %0d egress words (expect 18)", cap_cnt);
        if (cap_cnt !== 18) begin
            $display("  FAIL: expected 18 captured words, got %0d", cap_cnt);
            pass = 0;
        end

        check_word( 0, 8'hFF, EXP_HDR,    "NF2 header");
        check_word( 1, 8'h00, EXP_ETH1,   "Eth1 MAC swap");
        check_word( 2, 8'h00, EXP_ETH2,   "Eth2 MAC swap");
        check_word( 3, 8'h00, PKT_IPHDR1, "IP hdr1 (unchanged)");
        check_word( 4, 8'h00, EXP_IPHDR2, "IP hdr2 + IP swap");
        check_word( 5, 8'h00, EXP_UDPHDR, "UDP hdr + port swap");
        check_word( 6, 8'h00, EXP_UDP_CK, "UDP cksum + pad cleared");
        check_word( 7, 8'h00, EXP_LOGIT,  "logit0 replicated");
        check_word( 8, 8'h00, EXP_LOGIT,  "logit1 replicated");
        // BRAM[9..17] were not touched by orchestrator -- still carry the
        // original 1.0 feature payload from ingress, which is acceptable
        // (receiver only reads logits at word 7..8).
        check_word( 9, 8'h00, FEAT_1P0,   "residual feature payload");
        check_word(17, 8'h00, FEAT_1P0,   "residual feature payload (last)");

        // ===========================================================
        $display("\n###############################################");
        $display("# Test 2: 3-packet stress (auto-clear, mode flips, back-to-back)");
        $display("###############################################");

        for (k = 0; k < 3; k = k + 1) begin
            $display("\n  --- iteration %0d ---", k);
            cap_clear;
            inject_packet_1p0;
            repeat (8000) @(posedge clk);

            if (cap_cnt !== 18) begin
                $display("    FAIL: iter %0d captured %0d words (expected 18)",
                         k, cap_cnt);
                pass = 0;
            end else begin
                check_word(0, 8'hFF, EXP_HDR,   "NF2 header");
                check_word(1, 8'h00, EXP_ETH1,  "Eth1 MAC swap");
                check_word(7, 8'h00, EXP_LOGIT, "logit0");
                check_word(8, 8'h00, EXP_LOGIT, "logit1");
            end
        end

        // ===========================================================
        $display("\n=== Result ===");
        if (pass) $display("ALL TESTS PASSED");
        else      $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
