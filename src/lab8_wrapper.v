// lab8_wrapper — NF2.1 data path wrapper (FIFO + ARM CPU + GPU + DMA)
`timescale 1ns/1ps
`include "lab8_reg_defines.vh"

module lab8_wrapper
   #(
      parameter DATA_WIDTH = 64,
      parameter CTRL_WIDTH = DATA_WIDTH/8,
      parameter UDP_REG_SRC_WIDTH = 2
   )
   (
      // NF2.1 data path
      input  [DATA_WIDTH-1:0]             in_data,
      input  [CTRL_WIDTH-1:0]             in_ctrl,
      input                               in_wr,
      output                              in_rdy,

      output [DATA_WIDTH-1:0]             out_data,
      output [CTRL_WIDTH-1:0]             out_ctrl,
      output                              out_wr,
      input                               out_rdy,

      // Register ring
      input                               reg_req_in,
      input                               reg_ack_in,
      input                               reg_rd_wr_L_in,
      input  [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_in,
      input  [`CPCI_NF2_DATA_WIDTH-1:0]   reg_data_in,
      input  [UDP_REG_SRC_WIDTH-1:0]      reg_src_in,

      output                                  reg_req_out,
      output                                  reg_ack_out,
      output                                  reg_rd_wr_L_out,
      output [`UDP_REG_ADDR_WIDTH-1:0]        reg_addr_out,
      output [`CPCI_NF2_DATA_WIDTH-1:0]       reg_data_out,
      output [UDP_REG_SRC_WIDTH-1:0]          reg_src_out,

      input                                clk,
      input                                reset
   );

   // --- Address decode ---
   wire block_match = (reg_addr_in[`UDP_REG_ADDR_WIDTH-1:`LAB8_REG_ADDR_WIDTH]
                        == `LAB8_BLOCK_ADDR);

   wire [`LAB8_REG_ADDR_WIDTH-1:0] local_offset =
        reg_addr_in[`LAB8_REG_ADDR_WIDTH-1:0];

   // --- Convertible FIFO ---

   wire soft_reset;

   reg  [1:0]  fifo_mode;
   reg         fifo_drain_start;

   reg  [7:0]  fifo_proc_addr;
   reg  [71:0] fifo_proc_din;
   reg         fifo_proc_we;
   wire [71:0] fifo_proc_dout;

   wire        fifo_pkt_ready;
   wire        fifo_full;
   wire        fifo_empty;
   wire [7:0]  fifo_pkt_word_cnt;
   wire [7:0]  fifo_head_rd;
   wire [7:0]  fifo_tail_rd;

   convertible_fifo u_fifo (
      .clk           (clk),
      .reset         (soft_reset),
      .in_data       (in_data),
      .in_ctrl       (in_ctrl),
      .in_wr         (in_wr),
      .in_rdy        (in_rdy),
      .out_data      (out_data),
      .out_ctrl      (out_ctrl),
      .out_wr        (out_wr),
      .out_rdy       (out_rdy),
      .proc_addr     (fifo_proc_addr),
      .proc_din      (fifo_proc_din),
      .proc_we       (fifo_proc_we),
      .proc_dout     (fifo_proc_dout),
      .mode          (fifo_mode),
      .drain_start   (fifo_drain_start),
      .pkt_ready     (fifo_pkt_ready),
      .fifo_full     (fifo_full),
      .fifo_empty    (fifo_empty),
      .pkt_word_cnt  (fifo_pkt_word_cnt),
      .head_wr_val   (8'd0),
      .head_wr_en    (1'b0),
      .tail_wr_val   (8'd0),
      .tail_wr_en    (1'b0),
      .head_rd       (fifo_head_rd),
      .tail_rd       (fifo_tail_rd)
   );

   // --- ARM CPU ---

   wire        cpu_running;
   wire        cpu_reset;

   wire [7:0]  cpu_ext_mem_addr;
   wire [63:0] cpu_ext_mem_din;
   wire        cpu_ext_mem_we;
   wire [63:0] cpu_ext_mem_dout;

   wire        cpu_imem_wen;
   wire [8:0]  cpu_imem_addr;
   wire [31:0] cpu_imem_wdata;
   wire [31:0] cpu_imem_rdata;

   wire        cpu_dmem_wen;
   wire [7:0]  cpu_dmem_addr;
   wire [63:0] cpu_dmem_wdata;
   wire [63:0] cpu_dmem_rdata;

   wire [10:0] cpu_la_addr;
   wire [63:0] cpu_la_data;

   wire [3:0]  cpu_cpsr_0, cpu_cpsr_1, cpu_cpsr_2, cpu_cpsr_3;
   wire [1:0]  cpu_thread_id;
   wire        cpu_all_halted;

   processor u_cpu (
      .clk            (clk),
      .rst            (cpu_reset),
      .start_stop     (cpu_running),
      .intf_wen_imem  (cpu_imem_wen),
      .intf_addr_imem (cpu_imem_addr),
      .intf_data_imem (cpu_imem_wdata),
      .intf_dout_imem (cpu_imem_rdata),
      .web            (cpu_dmem_wen),
      .dinb           (cpu_dmem_wdata),
      .addrb          (cpu_dmem_addr),
      .doutb          (cpu_dmem_rdata),
      .ext_mem_addr   (cpu_ext_mem_addr),
      .ext_mem_din    (cpu_ext_mem_din),
      .ext_mem_we     (cpu_ext_mem_we),
      .ext_mem_dout   (cpu_ext_mem_dout),
      .la_rd_addr     (cpu_la_addr),
      .la_rd_data     (cpu_la_data),
      .dbg_cpsr_0     (cpu_cpsr_0),
      .dbg_cpsr_1     (cpu_cpsr_1),
      .dbg_cpsr_2     (cpu_cpsr_2),
      .dbg_cpsr_3     (cpu_cpsr_3),
      .dbg_thread_id  (cpu_thread_id),
      .all_halted     (cpu_all_halted)
   );

   assign cpu_ext_mem_dout = fifo_proc_dout[63:0];

   // --- GPU ---

   wire        gpu_reset;
   wire        gpu_kernel_done;

   wire [9:0]  gpu_imem_addr;
   wire        gpu_imem_wr_en;
   wire [31:0] gpu_imem_wr_data;
   wire [31:0] gpu_imem_rd_data;

   reg  [9:0]  gpu_dmem_addr;
   reg         gpu_dmem_wr_en;
   reg  [63:0] gpu_dmem_wr_data;
   wire [63:0] gpu_dmem_rd_data;

   gpu_top u_gpu (
      .clk             (clk),
      .rst             (gpu_reset),
      .thread_id       (32'd0),
      .block_id        (32'd0),
      .block_dim       (32'd1),
      .kernel_done     (gpu_kernel_done),
      .ext_imem_addr   (gpu_imem_addr),
      .ext_imem_wr_en  (gpu_imem_wr_en),
      .ext_imem_wr_data(gpu_imem_wr_data),
      .ext_imem_rd_data(gpu_imem_rd_data),
      .ext_dmem_addr   (gpu_dmem_addr),
      .ext_dmem_wr_en  (gpu_dmem_wr_en),
      .ext_dmem_wr_data(gpu_dmem_wr_data),
      .ext_dmem_rd_data(gpu_dmem_rd_data)
   );

   // --- DMA Engine ---

   localparam DMA_IDLE = 2'd0;
   localparam DMA_READ = 2'd1;
   localparam DMA_COPY = 2'd2;

   reg [1:0]  dma_state;
   reg        dma_dir;        // 0=FIFO->GPU, 1=GPU->FIFO
   reg [7:0]  dma_fifo_base;
   reg [9:0]  dma_gpu_base;
   reg [7:0]  dma_len;
   reg [7:0]  dma_rd_ptr;
   reg [7:0]  dma_wr_ptr;

   wire dma_active = (dma_state != DMA_IDLE);

   reg [31:0] reg_dma_ctrl;
   reg [31:0] reg_dma_fifo_addr;
   reg [31:0] reg_dma_gpu_addr;
   reg [31:0] reg_dma_length;

   wire dma_start;

   always @(posedge clk) begin
      if (reset) begin
         dma_state     <= DMA_IDLE;
         dma_dir       <= 1'b0;
         dma_fifo_base <= 8'd0;
         dma_gpu_base  <= 10'd0;
         dma_len       <= 8'd0;
         dma_rd_ptr    <= 8'd0;
         dma_wr_ptr    <= 8'd0;
      end else begin
         case (dma_state)
            DMA_IDLE: begin
               if (dma_start) begin
                  dma_dir       <= reg_dma_ctrl[1];
                  dma_fifo_base <= reg_dma_fifo_addr[7:0];
                  dma_gpu_base  <= reg_dma_gpu_addr[9:0];
                  dma_len       <= reg_dma_length[7:0];
                  dma_rd_ptr    <= 8'd0;
                  dma_wr_ptr    <= 8'd0;
                  dma_state     <= DMA_READ;
               end
            end
            DMA_READ: begin
               dma_rd_ptr <= 8'd1;
               dma_state  <= DMA_COPY;
            end
            DMA_COPY: begin
               dma_wr_ptr <= dma_wr_ptr + 8'd1;
               dma_rd_ptr <= dma_rd_ptr + 8'd1;
               if (dma_wr_ptr + 8'd1 == dma_len)
                  dma_state <= DMA_IDLE;
            end
            default: dma_state <= DMA_IDLE;
         endcase
      end
   end

   // --- PCI write strobes ---

   wire pci_wr_hit = reg_req_in && !reg_ack_in && block_match && !reg_rd_wr_L_in;

   reg bram_pci_wr_strobe;
   reg gpu_dmem_pci_wr_strobe;

   always @(posedge clk) begin
      if (reset) begin
         bram_pci_wr_strobe     <= 1'b0;
         gpu_dmem_pci_wr_strobe <= 1'b0;
      end else begin
         bram_pci_wr_strobe     <= pci_wr_hit
                                && (local_offset == `LAB8_BRAM_CMD)
                                && reg_data_in[0];
         gpu_dmem_pci_wr_strobe <= pci_wr_hit
                                && (local_offset == `LAB8_GPU_DMEM_CMD)
                                && reg_data_in[0];
      end
   end

   // --- Software-writable registers ---
   reg [31:0] reg_ctrl;
   reg [31:0] reg_fifo_mode_r;
   reg [31:0] reg_fifo_drain;
   reg [31:0] reg_bram_addr;
   reg [31:0] reg_bram_wd_lo;
   reg [31:0] reg_bram_wd_hi;
   reg [31:0] reg_bram_wctrl;
   reg [31:0] reg_bram_cmd;

   reg [31:0] reg_cpu_ctrl;
   reg [31:0] reg_cpu_imem_addr;
   reg [31:0] reg_cpu_imem_wdata;
   reg [31:0] reg_cpu_imem_cmd;
   reg [31:0] reg_cpu_dmem_addr;
   reg [31:0] reg_cpu_dmem_wd_lo;
   reg [31:0] reg_cpu_dmem_wd_hi;
   reg [31:0] reg_cpu_dmem_cmd;
   reg [31:0] reg_cpu_la_addr;

   reg [31:0] reg_gpu_ctrl;
   reg [31:0] reg_gpu_imem_addr;
   reg [31:0] reg_gpu_imem_wdata;
   reg [31:0] reg_gpu_imem_cmd;
   reg [31:0] reg_gpu_dmem_addr;
   reg [31:0] reg_gpu_dmem_wd_lo;
   reg [31:0] reg_gpu_dmem_wd_hi;
   reg [31:0] reg_gpu_dmem_cmd;
   assign soft_reset  = reset | reg_ctrl[0];
   assign cpu_running = reg_cpu_ctrl[0];
   assign cpu_reset   = reset | reg_cpu_ctrl[1];
   assign gpu_reset   = reset | reg_gpu_ctrl[0];

   assign cpu_imem_wen   = reg_cpu_imem_cmd[0];
   assign cpu_imem_addr  = reg_cpu_imem_addr[8:0];
   assign cpu_imem_wdata = reg_cpu_imem_wdata;

   assign cpu_dmem_wen   = reg_cpu_dmem_cmd[0];
   assign cpu_dmem_addr  = reg_cpu_dmem_addr[7:0];
   assign cpu_dmem_wdata = {reg_cpu_dmem_wd_hi, reg_cpu_dmem_wd_lo};

   assign cpu_la_addr = reg_cpu_la_addr[10:0];

   assign gpu_imem_addr    = reg_gpu_imem_addr[9:0];
   assign gpu_imem_wr_en   = reg_gpu_imem_cmd[0];
   assign gpu_imem_wr_data = reg_gpu_imem_wdata;

   assign dma_start = reg_dma_ctrl[0]
                    && (reg_fifo_mode_r[1:0] == 2'd2)
                    && (reg_dma_length[7:0] != 8'd0);

   // --- FIFO proc port mux ---
   wire cpu_drives_proc = (reg_fifo_mode_r[1:0] == 2'd1) && cpu_running;
   wire dma_drives_proc = (reg_fifo_mode_r[1:0] == 2'd2) && dma_active;
   always @(*) begin
      fifo_mode        = reg_fifo_mode_r[1:0];
      fifo_drain_start = reg_fifo_drain[0];

      if (cpu_drives_proc) begin
         fifo_proc_addr = cpu_ext_mem_addr;
         fifo_proc_din  = {8'h00, cpu_ext_mem_din};
         fifo_proc_we   = cpu_ext_mem_we;
      end else if (dma_drives_proc) begin
         if (!dma_dir) begin
            fifo_proc_addr = dma_fifo_base + dma_rd_ptr;
            fifo_proc_din  = 72'd0;
            fifo_proc_we   = 1'b0;
         end else begin
            fifo_proc_addr = dma_fifo_base + dma_wr_ptr;
            fifo_proc_din  = {8'h00, gpu_dmem_rd_data};
            fifo_proc_we   = (dma_state == DMA_COPY);
         end
      end else begin
         fifo_proc_addr = reg_bram_addr[7:0];
         fifo_proc_din  = {reg_bram_wctrl[7:0], reg_bram_wd_hi, reg_bram_wd_lo};
         fifo_proc_we   = bram_pci_wr_strobe;
      end
   end

   // --- GPU DMEM ext port mux ---
   always @(*) begin
      if (dma_active) begin
         if (!dma_dir) begin
            gpu_dmem_addr    = dma_gpu_base + {2'b0, dma_wr_ptr};
            gpu_dmem_wr_en   = (dma_state == DMA_COPY);
            gpu_dmem_wr_data = fifo_proc_dout[63:0];
         end else begin
            gpu_dmem_addr    = dma_gpu_base + {2'b0, dma_rd_ptr};
            gpu_dmem_wr_en   = 1'b0;
            gpu_dmem_wr_data = 64'd0;
         end
      end else begin
         gpu_dmem_addr    = reg_gpu_dmem_addr[9:0];
         gpu_dmem_wr_en   = gpu_dmem_pci_wr_strobe;
         gpu_dmem_wr_data = {reg_gpu_dmem_wd_hi, reg_gpu_dmem_wd_lo};
      end
   end

   // --- Hardware-readable registers ---

   wire [31:0] reg_status = {fifo_pkt_word_cnt, fifo_head_rd, fifo_tail_rd,
                             5'd0, fifo_empty, fifo_full, fifo_pkt_ready};

   wire [31:0] reg_bram_rd_lo  = fifo_proc_dout[31:0];
   wire [31:0] reg_bram_rd_hi  = fifo_proc_dout[63:32];
   wire [31:0] reg_bram_rctrl  = {24'd0, fifo_proc_dout[71:64]};

   wire [31:0] reg_cpu_status = {cpu_all_halted,
                                  3'd0,
                                  cpu_cpsr_3,
                                  cpu_cpsr_2,
                                  cpu_cpsr_1,
                                  cpu_cpsr_0,
                                  cpu_thread_id,
                                  6'd0,
                                  2'd0,
                                  cpu_running,
                                  ~cpu_reset};

   wire [31:0] reg_cpu_imem_rdata  = cpu_imem_rdata;
   wire [31:0] reg_cpu_dmem_rd_lo  = cpu_dmem_rdata[31:0];
   wire [31:0] reg_cpu_dmem_rd_hi  = cpu_dmem_rdata[63:32];
   wire [31:0] reg_cpu_la_rd_lo    = cpu_la_data[31:0];
   wire [31:0] reg_cpu_la_rd_hi    = cpu_la_data[63:32];

   wire [31:0] reg_gpu_status = {30'd0, dma_active, gpu_kernel_done};

   // --- GPU cycle counter ---
   // Counts core_clk cycles from gpu_reset deassertion until kernel_done
   // first asserts. Matches the cycle count reported by tb_ids_inference
   // and tb_ids_bcast, enabling scalar-vs-BCAST speedup measurement directly
   // from the host via PCI read.
   reg [31:0] gpu_cycle_count;
   reg        gpu_count_armed;

   always @(posedge clk) begin
      if (reset || gpu_reset) begin
         gpu_cycle_count <= 32'd0;
         gpu_count_armed <= 1'b1;
      end else if (gpu_count_armed) begin
         gpu_cycle_count <= gpu_cycle_count + 32'd1;
         if (gpu_kernel_done)
            gpu_count_armed <= 1'b0;
      end
   end

   wire [31:0] reg_gpu_cycle_count = gpu_cycle_count;

   wire [31:0] reg_gpu_imem_rdata  = gpu_imem_rd_data;
   wire [31:0] reg_gpu_dmem_rd_lo  = gpu_dmem_rd_data[31:0];
   wire [31:0] reg_gpu_dmem_rd_hi  = gpu_dmem_rd_data[63:32];

   // --- Register ring state machine ---

   reg                              reg_req_held;
   reg                              reg_ack_held;
   reg                              reg_rd_wr_L_held;
   reg [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_held;
   reg [`CPCI_NF2_DATA_WIDTH-1:0]   reg_rd_data;
   reg [`CPCI_NF2_DATA_WIDTH-1:0]   reg_rd_data_mux;
   reg [UDP_REG_SRC_WIDTH-1:0]      reg_src_held;

   // Combinational readback mux.  Kept in its own always @* block so XST
   // does not put it into the large clocked write-path always block
   // (doing so caused reg_rd_data to be eliminated in the
   // 2026-04-11 synth, which made every PCI read return 0xDEADBEEF).
   always @* begin
      case (reg_addr_in[`LAB8_REG_ADDR_WIDTH-1:0])
         `LAB8_CTRL:              reg_rd_data_mux = reg_ctrl;
         `LAB8_STATUS:            reg_rd_data_mux = reg_status;
         `LAB8_FIFO_MODE:         reg_rd_data_mux = reg_fifo_mode_r;
         `LAB8_FIFO_DRAIN:        reg_rd_data_mux = reg_fifo_drain;
         `LAB8_BRAM_ADDR:         reg_rd_data_mux = reg_bram_addr;
         `LAB8_BRAM_WD_LO:        reg_rd_data_mux = reg_bram_wd_lo;
         `LAB8_BRAM_WD_HI:        reg_rd_data_mux = reg_bram_wd_hi;
         `LAB8_BRAM_WCTRL:        reg_rd_data_mux = reg_bram_wctrl;
         `LAB8_BRAM_CMD:          reg_rd_data_mux = reg_bram_cmd;
         `LAB8_BRAM_RD_LO:        reg_rd_data_mux = reg_bram_rd_lo;
         `LAB8_BRAM_RD_HI:        reg_rd_data_mux = reg_bram_rd_hi;
         `LAB8_BRAM_RCTRL:        reg_rd_data_mux = reg_bram_rctrl;
         `LAB8_CPU_CTRL:          reg_rd_data_mux = reg_cpu_ctrl;
         `LAB8_CPU_STATUS:        reg_rd_data_mux = reg_cpu_status;
         `LAB8_CPU_IMEM_ADDR:     reg_rd_data_mux = reg_cpu_imem_addr;
         `LAB8_CPU_IMEM_WDATA:    reg_rd_data_mux = reg_cpu_imem_wdata;
         `LAB8_CPU_IMEM_CMD:      reg_rd_data_mux = reg_cpu_imem_cmd;
         `LAB8_CPU_IMEM_RDATA:    reg_rd_data_mux = reg_cpu_imem_rdata;
         `LAB8_CPU_DMEM_ADDR:     reg_rd_data_mux = reg_cpu_dmem_addr;
         `LAB8_CPU_DMEM_WDATA_LO: reg_rd_data_mux = reg_cpu_dmem_wd_lo;
         `LAB8_CPU_DMEM_WDATA_HI: reg_rd_data_mux = reg_cpu_dmem_wd_hi;
         `LAB8_CPU_DMEM_CMD:      reg_rd_data_mux = reg_cpu_dmem_cmd;
         `LAB8_CPU_DMEM_RDATA_LO: reg_rd_data_mux = reg_cpu_dmem_rd_lo;
         `LAB8_CPU_DMEM_RDATA_HI: reg_rd_data_mux = reg_cpu_dmem_rd_hi;
         `LAB8_CPU_LA_ADDR:       reg_rd_data_mux = reg_cpu_la_addr;
         `LAB8_CPU_LA_RDATA_LO:   reg_rd_data_mux = reg_cpu_la_rd_lo;
         `LAB8_CPU_LA_RDATA_HI:   reg_rd_data_mux = reg_cpu_la_rd_hi;
         `LAB8_GPU_CTRL:          reg_rd_data_mux = reg_gpu_ctrl;
         `LAB8_GPU_STATUS:        reg_rd_data_mux = reg_gpu_status;
         `LAB8_GPU_IMEM_ADDR:     reg_rd_data_mux = reg_gpu_imem_addr;
         `LAB8_GPU_IMEM_WDATA:    reg_rd_data_mux = reg_gpu_imem_wdata;
         `LAB8_GPU_IMEM_CMD:      reg_rd_data_mux = reg_gpu_imem_cmd;
         `LAB8_GPU_IMEM_RDATA:    reg_rd_data_mux = reg_gpu_imem_rdata;
         `LAB8_GPU_DMEM_ADDR:     reg_rd_data_mux = reg_gpu_dmem_addr;
         `LAB8_GPU_DMEM_WDATA_LO: reg_rd_data_mux = reg_gpu_dmem_wd_lo;
         `LAB8_GPU_DMEM_WDATA_HI: reg_rd_data_mux = reg_gpu_dmem_wd_hi;
         `LAB8_GPU_DMEM_CMD:      reg_rd_data_mux = reg_gpu_dmem_cmd;
         `LAB8_GPU_DMEM_RDATA_LO: reg_rd_data_mux = reg_gpu_dmem_rd_lo;
         `LAB8_GPU_DMEM_RDATA_HI: reg_rd_data_mux = reg_gpu_dmem_rd_hi;
         `LAB8_DMA_CTRL:          reg_rd_data_mux = reg_dma_ctrl;
         `LAB8_DMA_FIFO_ADDR:     reg_rd_data_mux = reg_dma_fifo_addr;
         `LAB8_DMA_GPU_ADDR:      reg_rd_data_mux = reg_dma_gpu_addr;
         `LAB8_DMA_LENGTH:        reg_rd_data_mux = reg_dma_length;
         `LAB8_GPU_CYCLE_COUNT:   reg_rd_data_mux = reg_gpu_cycle_count;
         default:                 reg_rd_data_mux = 32'hDEAD_BEEF;
      endcase
   end

   // Dedicated sequential flop for reg_rd_data, driven by the combinational
   // mux when we own the request, otherwise passing the upstream bus
   // through the NF2 register ring.
   always @(posedge clk) begin
      if (reset) begin
         reg_rd_data <= 32'd0;
      end
      else if (reg_req_in && !reg_ack_in && block_match && reg_rd_wr_L_in) begin
         reg_rd_data <= reg_rd_data_mux;
      end
      else begin
         reg_rd_data <= reg_data_in;
      end
   end

   always @(posedge clk) begin
      if (reset) begin
         reg_ctrl         <= 32'd0;
         reg_fifo_mode_r  <= 32'd0;
         reg_fifo_drain   <= 32'd0;
         reg_bram_addr    <= 32'd0;
         reg_bram_wd_lo   <= 32'd0;
         reg_bram_wd_hi   <= 32'd0;
         reg_bram_wctrl   <= 32'd0;
         reg_bram_cmd     <= 32'd0;

         reg_cpu_ctrl        <= 32'd0;
         reg_cpu_imem_addr   <= 32'd0;
         reg_cpu_imem_wdata  <= 32'd0;
         reg_cpu_imem_cmd    <= 32'd0;
         reg_cpu_dmem_addr   <= 32'd0;
         reg_cpu_dmem_wd_lo  <= 32'd0;
         reg_cpu_dmem_wd_hi  <= 32'd0;
         reg_cpu_dmem_cmd    <= 32'd0;
         reg_cpu_la_addr     <= 32'd0;

         reg_gpu_ctrl        <= 32'h1; // GPU starts in reset
         reg_gpu_imem_addr   <= 32'd0;
         reg_gpu_imem_wdata  <= 32'd0;
         reg_gpu_imem_cmd    <= 32'd0;
         reg_gpu_dmem_addr   <= 32'd0;
         reg_gpu_dmem_wd_lo  <= 32'd0;
         reg_gpu_dmem_wd_hi  <= 32'd0;
         reg_gpu_dmem_cmd    <= 32'd0;

         reg_dma_ctrl        <= 32'd0;
         reg_dma_fifo_addr   <= 32'd0;
         reg_dma_gpu_addr    <= 32'd0;
         reg_dma_length      <= 32'd0;

         reg_req_held     <= 1'b0;
         reg_ack_held     <= 1'b0;
         reg_rd_wr_L_held <= 1'b0;
         reg_addr_held    <= {`UDP_REG_ADDR_WIDTH{1'b0}};
         reg_src_held     <= {UDP_REG_SRC_WIDTH{1'b0}};
      end
      else begin
         reg_req_held     <= reg_req_in;
         reg_ack_held     <= reg_ack_in;
         reg_rd_wr_L_held <= reg_rd_wr_L_in;
         reg_addr_held    <= reg_addr_in;
         reg_src_held     <= reg_src_in;

         if (reg_req_in && !reg_ack_in && block_match) begin
            reg_ack_held <= 1'b1;

            if (!reg_rd_wr_L_in) begin
               case (local_offset)
                  `LAB8_CTRL:              reg_ctrl           <= reg_data_in;
                  `LAB8_FIFO_MODE:         reg_fifo_mode_r    <= reg_data_in;
                  `LAB8_FIFO_DRAIN:        reg_fifo_drain     <= reg_data_in;
                  `LAB8_BRAM_ADDR:         reg_bram_addr      <= reg_data_in;
                  `LAB8_BRAM_WD_LO:        reg_bram_wd_lo     <= reg_data_in;
                  `LAB8_BRAM_WD_HI:        reg_bram_wd_hi     <= reg_data_in;
                  `LAB8_BRAM_WCTRL:        reg_bram_wctrl     <= reg_data_in;
                  `LAB8_BRAM_CMD:          reg_bram_cmd       <= reg_data_in;
                  `LAB8_CPU_CTRL:          reg_cpu_ctrl       <= reg_data_in;
                  `LAB8_CPU_IMEM_ADDR:     reg_cpu_imem_addr  <= reg_data_in;
                  `LAB8_CPU_IMEM_WDATA:    reg_cpu_imem_wdata <= reg_data_in;
                  `LAB8_CPU_IMEM_CMD:      reg_cpu_imem_cmd   <= reg_data_in;
                  `LAB8_CPU_DMEM_ADDR:     reg_cpu_dmem_addr  <= reg_data_in;
                  `LAB8_CPU_DMEM_WDATA_LO: reg_cpu_dmem_wd_lo <= reg_data_in;
                  `LAB8_CPU_DMEM_WDATA_HI: reg_cpu_dmem_wd_hi <= reg_data_in;
                  `LAB8_CPU_DMEM_CMD:      reg_cpu_dmem_cmd   <= reg_data_in;
                  `LAB8_CPU_LA_ADDR:       reg_cpu_la_addr    <= reg_data_in;
                  `LAB8_GPU_CTRL:          reg_gpu_ctrl       <= reg_data_in;
                  `LAB8_GPU_IMEM_ADDR:     reg_gpu_imem_addr  <= reg_data_in;
                  `LAB8_GPU_IMEM_WDATA:    reg_gpu_imem_wdata <= reg_data_in;
                  `LAB8_GPU_IMEM_CMD:      reg_gpu_imem_cmd   <= reg_data_in;
                  `LAB8_GPU_DMEM_ADDR:     reg_gpu_dmem_addr <= reg_data_in;
                  `LAB8_GPU_DMEM_WDATA_LO: reg_gpu_dmem_wd_lo <= reg_data_in;
                  `LAB8_GPU_DMEM_WDATA_HI: reg_gpu_dmem_wd_hi <= reg_data_in;
                  `LAB8_GPU_DMEM_CMD:      reg_gpu_dmem_cmd   <= reg_data_in;
                  `LAB8_DMA_CTRL:          reg_dma_ctrl       <= reg_data_in;
                  `LAB8_DMA_FIFO_ADDR:     reg_dma_fifo_addr  <= reg_data_in;
                  `LAB8_DMA_GPU_ADDR:      reg_dma_gpu_addr   <= reg_data_in;
                  `LAB8_DMA_LENGTH:        reg_dma_length     <= reg_data_in;
                  default: ;
               endcase
            end
         end

         // Auto-clear strobes
         if (reg_fifo_drain[0])    reg_fifo_drain[0]    <= 1'b0;
         if (reg_bram_cmd[0])      reg_bram_cmd[0]      <= 1'b0;
         if (reg_cpu_imem_cmd[0])  reg_cpu_imem_cmd[0]  <= 1'b0;
         if (reg_cpu_dmem_cmd[0])  reg_cpu_dmem_cmd[0]  <= 1'b0;
         if (reg_gpu_imem_cmd[0])  reg_gpu_imem_cmd[0]  <= 1'b0;
         if (reg_gpu_dmem_cmd[0])  reg_gpu_dmem_cmd[0]  <= 1'b0;
         if (reg_dma_ctrl[0])      reg_dma_ctrl[0]      <= 1'b0;
      end
   end

   assign reg_req_out     = reg_req_held;
   assign reg_ack_out     = reg_ack_held;
   assign reg_rd_wr_L_out = reg_rd_wr_L_held;
   assign reg_addr_out    = reg_addr_held;
   assign reg_data_out    = reg_rd_data;
   assign reg_src_out     = reg_src_held;

endmodule
