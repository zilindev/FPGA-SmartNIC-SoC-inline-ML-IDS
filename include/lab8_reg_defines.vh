///////////////////////////////////////////////////////////////////////////////
// lab8_reg_defines.vh — Register definitions for Lab8 wrapper
//
// For standalone simulation (iverilog). On the NetFPGA VM, the NF2
// build system generates equivalent defines from XML (registers.v).
// All defines are guarded with `ifndef so the generated values take priority.
//
// Convention: register offsets are WORD-indexed (not byte-indexed).
// PCI byte address = BAR2 + (block_base + word_offset) * 4.
// This matches the NF2 register generator output (e.g. PROCESSOR_CTRL=6'h0).
///////////////////////////////////////////////////////////////////////////////

`ifndef _LAB8_REG_DEFINES_VH_
`define _LAB8_REG_DEFINES_VH_

// NF2 system defaults (only define if NF2 build hasn't provided them)
`ifndef UDP_REG_ADDR_WIDTH
`define UDP_REG_ADDR_WIDTH    23
`endif

`ifndef CPCI_NF2_DATA_WIDTH
`define CPCI_NF2_DATA_WIDTH   32
`endif

// Lab8 block address in NF2 register space
//
// blocksize = 256 bytes → 64 word slots → REG_ADDR_WIDTH = 6
// BLOCK_ADDR_WIDTH = UDP_REG_ADDR_WIDTH - REG_ADDR_WIDTH = 23 - 6 = 17
//
// BLOCK_ADDR placeholder for standalone simulation. The actual value is
// assigned by the NF2 register generator based on the memalloc order in
// project.xml. After first `make` on the VM, check the generated
// registers.v for the real LAB8_BLOCK_ADDR.
//
// Reference: Lab7 IN_ARB (blocksize=256) gets 17'h00010 on reference_nic.
//            Lab6 PROCESSOR (blocksize=256) gets 17'h00002 on reference_router.
`ifndef LAB8_REG_ADDR_WIDTH
`define LAB8_REG_ADDR_WIDTH   6
`endif

`ifndef LAB8_BLOCK_ADDR
`define LAB8_BLOCK_ADDR       17'h00011
`endif

// ---------------------------------------------------------------
// Register offsets (WORD-indexed, matching NF2 register generator)
//
// Word offset N corresponds to PCI byte offset N*4.
// Order must match lab8.xml register declaration order.
// All guarded with `ifndef so generated registers.v takes priority.
// ---------------------------------------------------------------

// FIFO control/status
`ifndef LAB8_CTRL
`define LAB8_CTRL             6'h00   // RW  bit[0]=soft_reset
`endif
`ifndef LAB8_STATUS
`define LAB8_STATUS           6'h01   // RO  {pkt_word_cnt, ..., empty, full, pkt_ready}
`endif
`ifndef LAB8_FIFO_MODE
`define LAB8_FIFO_MODE        6'h02   // RW  mode[1:0]
`endif
`ifndef LAB8_FIFO_DRAIN
`define LAB8_FIFO_DRAIN       6'h03   // RW  bit[0]=drain_start (auto-clear)
`endif

// FIFO BRAM access via PCI
`ifndef LAB8_BRAM_ADDR
`define LAB8_BRAM_ADDR        6'h04   // RW  BRAM address [7:0]
`endif
`ifndef LAB8_BRAM_WD_LO
`define LAB8_BRAM_WD_LO       6'h05   // RW  write data [31:0]
`endif
`ifndef LAB8_BRAM_WD_HI
`define LAB8_BRAM_WD_HI       6'h06   // RW  write data [63:32]
`endif
`ifndef LAB8_BRAM_WCTRL
`define LAB8_BRAM_WCTRL       6'h07   // RW  write ctrl [7:0]
`endif
`ifndef LAB8_BRAM_CMD
`define LAB8_BRAM_CMD         6'h08   // RW  bit[0]=write strobe (auto-clear)
`endif
`ifndef LAB8_BRAM_RD_LO
`define LAB8_BRAM_RD_LO       6'h09   // RO  read data [31:0]
`endif
`ifndef LAB8_BRAM_RD_HI
`define LAB8_BRAM_RD_HI       6'h0A   // RO  read data [63:32]
`endif
`ifndef LAB8_BRAM_RCTRL
`define LAB8_BRAM_RCTRL       6'h0B   // RO  read ctrl [7:0]
`endif

// CPU registers (Batch 3)  6'h0C - 6'h1A
`ifndef LAB8_CPU_CTRL
`define LAB8_CPU_CTRL          6'h0C   // RW  bit[0]=start_stop, bit[1]=cpu_reset
`endif
`ifndef LAB8_CPU_STATUS
`define LAB8_CPU_STATUS        6'h0D   // RO  {all_halted, cpsr_3..0, thread_id, ...}
`endif
`ifndef LAB8_CPU_IMEM_ADDR
`define LAB8_CPU_IMEM_ADDR     6'h0E   // RW  IMEM address [8:0]
`endif
`ifndef LAB8_CPU_IMEM_WDATA
`define LAB8_CPU_IMEM_WDATA    6'h0F   // RW  IMEM write data [31:0]
`endif
`ifndef LAB8_CPU_IMEM_CMD
`define LAB8_CPU_IMEM_CMD      6'h10   // RW  bit[0]=write strobe (auto-clear)
`endif
`ifndef LAB8_CPU_IMEM_RDATA
`define LAB8_CPU_IMEM_RDATA    6'h11   // RO  IMEM readback [31:0]
`endif
`ifndef LAB8_CPU_DMEM_ADDR
`define LAB8_CPU_DMEM_ADDR     6'h12   // RW  DMEM address [7:0]
`endif
`ifndef LAB8_CPU_DMEM_WDATA_LO
`define LAB8_CPU_DMEM_WDATA_LO 6'h13   // RW  DMEM write data [31:0]
`endif
`ifndef LAB8_CPU_DMEM_WDATA_HI
`define LAB8_CPU_DMEM_WDATA_HI 6'h14   // RW  DMEM write data [63:32]
`endif
`ifndef LAB8_CPU_DMEM_CMD
`define LAB8_CPU_DMEM_CMD      6'h15   // RW  bit[0]=write strobe (auto-clear)
`endif
`ifndef LAB8_CPU_DMEM_RDATA_LO
`define LAB8_CPU_DMEM_RDATA_LO 6'h16   // RO  DMEM readback [31:0]
`endif
`ifndef LAB8_CPU_DMEM_RDATA_HI
`define LAB8_CPU_DMEM_RDATA_HI 6'h17   // RO  DMEM readback [63:32]
`endif
`ifndef LAB8_CPU_LA_ADDR
`define LAB8_CPU_LA_ADDR       6'h18   // RW  Logic analyzer addr [10:0]
`endif
`ifndef LAB8_CPU_LA_RDATA_LO
`define LAB8_CPU_LA_RDATA_LO   6'h19   // RO  LA readback [31:0]
`endif
`ifndef LAB8_CPU_LA_RDATA_HI
`define LAB8_CPU_LA_RDATA_HI   6'h1A   // RO  LA readback [63:32]
`endif

// GPU registers (Batch 4)  6'h20 - 6'h2B
`ifndef LAB8_GPU_CTRL
`define LAB8_GPU_CTRL           6'h20   // RW  bit[0]=gpu_reset (1=hold)
`endif
`ifndef LAB8_GPU_STATUS
`define LAB8_GPU_STATUS         6'h21   // RO  bit[0]=kernel_done, bit[1]=dma_busy
`endif
`ifndef LAB8_GPU_IMEM_ADDR
`define LAB8_GPU_IMEM_ADDR      6'h22   // RW  IMEM address [9:0]
`endif
`ifndef LAB8_GPU_IMEM_WDATA
`define LAB8_GPU_IMEM_WDATA     6'h23   // RW  IMEM write data [31:0]
`endif
`ifndef LAB8_GPU_IMEM_CMD
`define LAB8_GPU_IMEM_CMD       6'h24   // RW  bit[0]=write strobe (auto-clear)
`endif
`ifndef LAB8_GPU_IMEM_RDATA
`define LAB8_GPU_IMEM_RDATA     6'h25   // RO  IMEM readback [31:0]
`endif
`ifndef LAB8_GPU_DMEM_ADDR
`define LAB8_GPU_DMEM_ADDR      6'h26   // RW  DMEM address [9:0]
`endif
`ifndef LAB8_GPU_DMEM_WDATA_LO
`define LAB8_GPU_DMEM_WDATA_LO  6'h27   // RW  DMEM write data [31:0]
`endif
`ifndef LAB8_GPU_DMEM_WDATA_HI
`define LAB8_GPU_DMEM_WDATA_HI  6'h28   // RW  DMEM write data [63:32]
`endif
`ifndef LAB8_GPU_DMEM_CMD
`define LAB8_GPU_DMEM_CMD       6'h29   // RW  bit[0]=write strobe (auto-clear)
`endif
`ifndef LAB8_GPU_DMEM_RDATA_LO
`define LAB8_GPU_DMEM_RDATA_LO  6'h2A   // RO  DMEM readback [31:0]
`endif
`ifndef LAB8_GPU_DMEM_RDATA_HI
`define LAB8_GPU_DMEM_RDATA_HI  6'h2B   // RO  DMEM readback [63:32]
`endif

// DMA registers (Batch 4)  6'h2C - 6'h2F
`ifndef LAB8_DMA_CTRL
`define LAB8_DMA_CTRL           6'h2C   // RW  bit[0]=start (auto-clear), bit[1]=dir (0=FIFO→GPU, 1=GPU→FIFO)
`endif
`ifndef LAB8_DMA_FIFO_ADDR
`define LAB8_DMA_FIFO_ADDR      6'h2D   // RW  FIFO BRAM start address [7:0]
`endif
`ifndef LAB8_DMA_GPU_ADDR
`define LAB8_DMA_GPU_ADDR       6'h2E   // RW  GPU DMEM start address [9:0]
`endif
`ifndef LAB8_DMA_LENGTH
`define LAB8_DMA_LENGTH          6'h2F   // RW  word count [7:0]
`endif

// GPU performance counter (Lab 10)
`ifndef LAB8_GPU_CYCLE_COUNT
`define LAB8_GPU_CYCLE_COUNT     6'h30   // RO  GPU cycles from reset-release to kernel_done
`endif

`endif
