// tb_ids_bcast_trained.v -- Run BCAST-optimized IDS kernel with any data file
// and print logit outputs. Uses trained-weights DMEM by default.
//
// Usage: override data path via plusarg +DMEM_FILE=path or just edit readmemh.

`timescale 1ns / 1ps

module tb_ids_bcast_trained;

    parameter CLK_PERIOD  = 8;
    parameter IMEM_WORDS  = 108;
    parameter DMEM_WORDS  = 128;
    parameter MAX_CYCLES  = 10000;

    parameter H1_BASE  = 59;
    parameter H2_BASE  = 109;
    parameter OUT_BASE = 126;

    reg clk, rst;
    always #(CLK_PERIOD/2) clk = ~clk;

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
        .clk(clk), .rst(rst),
        .thread_id(32'd0), .block_id(32'd0), .block_dim(32'd1),
        .kernel_done(kernel_done),
        .ext_imem_addr(ext_imem_addr),
        .ext_imem_wr_en(ext_imem_wr_en),
        .ext_imem_wr_data(ext_imem_wr_data),
        .ext_imem_rd_data(ext_imem_rd_data),
        .ext_dmem_addr(ext_dmem_addr),
        .ext_dmem_wr_en(ext_dmem_wr_en),
        .ext_dmem_wr_data(ext_dmem_wr_data),
        .ext_dmem_rd_data(ext_dmem_rd_data)
    );

    reg [31:0] imem_file [0:1023];
    reg [63:0] dmem_file [0:1023];

    integer i, cycle_count;
    reg [63:0] read_val;

    task write_imem;
        input [9:0] addr; input [31:0] data;
        begin
            @(posedge clk);
            ext_imem_addr = addr; ext_imem_wr_data = data; ext_imem_wr_en = 1;
            @(posedge clk);
            ext_imem_wr_en = 0;
        end
    endtask

    task write_dmem;
        input [9:0] addr; input [63:0] data;
        begin
            @(posedge clk);
            ext_dmem_addr = addr; ext_dmem_wr_data = data; ext_dmem_wr_en = 1;
            @(posedge clk);
            ext_dmem_wr_en = 0;
        end
    endtask

    task read_dmem;
        input [9:0] addr; output [63:0] data;
        begin
            @(posedge clk);
            ext_dmem_addr = addr; ext_dmem_wr_en = 0;
            @(posedge clk);
            data = ext_dmem_rd_data;
        end
    endtask

    initial begin
        for (i = 0; i < 1024; i = i + 1) begin
            imem_file[i] = 32'h0;
            dmem_file[i] = 64'h0;
        end

        $readmemh("programs/gpu/ann_ids_11_16_8_2_bcast.hex", imem_file);
        $readmemh("programs/gpu/data_ids_trained_bcast.hex", dmem_file);

        clk = 0; rst = 1;
        ext_imem_wr_en = 0; ext_dmem_wr_en = 0;
        ext_imem_addr = 0;  ext_dmem_addr = 0;
        ext_imem_wr_data = 0; ext_dmem_wr_data = 0;
        repeat (5) @(posedge clk);

        $display("============================================================");
        $display("IDS MLP BCAST - trained weights test");
        $display("============================================================");

        for (i = 0; i < IMEM_WORDS; i = i + 1) write_imem(i[9:0], imem_file[i]);
        for (i = 0; i < DMEM_WORDS; i = i + 1) write_dmem(i[9:0], dmem_file[i]);

        repeat (2) @(posedge clk);
        rst = 0;
        cycle_count = 0;
        while (!kernel_done && cycle_count < MAX_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        rst = 1;
        repeat (2) @(posedge clk);

        $display("GPU halted after %0d cycles", cycle_count);

        $display("\n--- Layer 1 outputs (h1[0..15]) ---");
        for (i = 0; i < 16; i = i + 1) begin
            read_dmem(H1_BASE + i, read_val);
            $display("  h1[%2d] @DMEM[%0d] = 0x%016h", i, H1_BASE+i, read_val);
        end

        $display("\n--- Layer 2 outputs (h2[0..7]) ---");
        for (i = 0; i < 8; i = i + 1) begin
            read_dmem(H2_BASE + i, read_val);
            $display("  h2[%0d] @DMEM[%0d] = 0x%016h", i, H2_BASE+i, read_val);
        end

        $display("\n--- Layer 3 outputs (logits) ---");
        read_dmem(OUT_BASE, read_val);
        $display("  out[0] @DMEM[%0d] = 0x%016h", OUT_BASE, read_val);
        read_dmem(OUT_BASE + 1, read_val);
        $display("  out[1] @DMEM[%0d] = 0x%016h", OUT_BASE+1, read_val);

        #100; $finish;
    end

endmodule
