#!/usr/bin/env python
# -*- coding: ascii -*-
#
# lab9reg.py -- Lab9 Network Processor register interface for NetFPGA NF2.1
#
# Forked from lab8reg.py for Lab 9 (subgroup integration phase).
# Includes all FIFO, CPU, GPU, and DMA commands plus the GPU standalone
# test suite (5 CUDA kernels + 2 ANN inference tests) and real network
# packet ANN tests.
#
# Compatible with Python 2.4+ and Python 3.x.
# Wraps the NF2 regread/regwrite CLI tools.
#
# Usage:  python lab9reg.py <command> [args...]
#
# Register map (word offsets from block base):
#   0x00-0x0B  FIFO registers (CTRL, STATUS, MODE, DRAIN, BRAM)
#   0x0C-0x1A  CPU registers (CTRL, STATUS, IMEM, DMEM, LA)
#   0x1B-0x1F  (reserved padding)
#   0x20-0x2B  GPU registers (CTRL, STATUS, IMEM, DMEM)
#   0x2C-0x2F  DMA registers (CTRL, FIFO_ADDR, GPU_ADDR, LENGTH)

import subprocess
import sys
import os
import re
import time

# ============================================================================
# Block base address
# ============================================================================
# LAB8_BLOCK_ADDR = 17'h00011.  reg_addr = 17'h00011 << 6 = 23'h000440.
# PCI byte addr = BAR2 + 0x440 * 4 = 0x2001100.
# Verify with:  grep LAB8_BLOCK_ADDR include/registers.v  (on the VM)
BASE = 0x2001100

# --- FIFO registers ---
CTRL_REG       = BASE + 0x00 * 4
STATUS_REG     = BASE + 0x01 * 4
FIFO_MODE_REG  = BASE + 0x02 * 4
FIFO_DRAIN_REG = BASE + 0x03 * 4
BRAM_ADDR_REG  = BASE + 0x04 * 4
BRAM_WD_LO_REG = BASE + 0x05 * 4
BRAM_WD_HI_REG = BASE + 0x06 * 4
BRAM_WCTRL_REG = BASE + 0x07 * 4
BRAM_CMD_REG   = BASE + 0x08 * 4
BRAM_RD_LO_REG = BASE + 0x09 * 4
BRAM_RD_HI_REG = BASE + 0x0A * 4
BRAM_RCTRL_REG = BASE + 0x0B * 4

# --- CPU registers ---
CPU_CTRL_REG       = BASE + 0x0C * 4
CPU_STATUS_REG     = BASE + 0x0D * 4
CPU_IMEM_ADDR_REG  = BASE + 0x0E * 4
CPU_IMEM_WDATA_REG = BASE + 0x0F * 4
CPU_IMEM_CMD_REG   = BASE + 0x10 * 4
CPU_IMEM_RDATA_REG = BASE + 0x11 * 4
CPU_DMEM_ADDR_REG  = BASE + 0x12 * 4
CPU_DMEM_WD_LO_REG = BASE + 0x13 * 4
CPU_DMEM_WD_HI_REG = BASE + 0x14 * 4
CPU_DMEM_CMD_REG   = BASE + 0x15 * 4
CPU_DMEM_RD_LO_REG = BASE + 0x16 * 4
CPU_DMEM_RD_HI_REG = BASE + 0x17 * 4
CPU_LA_ADDR_REG    = BASE + 0x18 * 4
CPU_LA_RD_LO_REG   = BASE + 0x19 * 4
CPU_LA_RD_HI_REG   = BASE + 0x1A * 4

# --- GPU registers ---
GPU_CTRL_REG       = BASE + 0x20 * 4
GPU_STATUS_REG     = BASE + 0x21 * 4
GPU_IMEM_ADDR_REG  = BASE + 0x22 * 4
GPU_IMEM_WDATA_REG = BASE + 0x23 * 4
GPU_IMEM_CMD_REG   = BASE + 0x24 * 4
GPU_IMEM_RDATA_REG = BASE + 0x25 * 4
GPU_DMEM_ADDR_REG  = BASE + 0x26 * 4
GPU_DMEM_WD_LO_REG = BASE + 0x27 * 4
GPU_DMEM_WD_HI_REG = BASE + 0x28 * 4
GPU_DMEM_CMD_REG   = BASE + 0x29 * 4
GPU_DMEM_RD_LO_REG = BASE + 0x2A * 4
GPU_DMEM_RD_HI_REG = BASE + 0x2B * 4

# --- DMA registers ---
DMA_CTRL_REG      = BASE + 0x2C * 4
DMA_FIFO_ADDR_REG = BASE + 0x2D * 4
DMA_GPU_ADDR_REG  = BASE + 0x2E * 4
DMA_LENGTH_REG    = BASE + 0x2F * 4

# --- GPU performance counter (Lab 10) ---
GPU_CYCLE_COUNT_REG = BASE + 0x30 * 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PORT_NAMES = {
    0x01: "MAC0/nf2c0", 0x02: "CPU0", 0x04: "MAC1/nf2c1", 0x08: "CPU1",
    0x10: "MAC2/nf2c2", 0x20: "CPU2", 0x40: "MAC3/nf2c3", 0x80: "CPU3",
}

# ============================================================================
# Low-level register access
# ============================================================================

def regwrite(addr, value):
    cmd = "regwrite 0x%07x 0x%08x" % (addr, value)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()

def regread(addr):
    cmd = "regread 0x%07x" % addr
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, _ = p.communicate()
    if not isinstance(out, str):
        out = out.decode("utf-8", "replace")
    m = re.search(r'Reg\s+\S+\s+\(\d+\):\s+(0x[0-9a-fA-F]+)', out)
    if m:
        return int(m.group(1), 16)
    return 0

# ============================================================================
# FIFO helpers
# ============================================================================

def fifo_reset():
    regwrite(CTRL_REG, 0x1)
    regwrite(CTRL_REG, 0x0)

def fifo_set_mode(mode):
    regwrite(FIFO_MODE_REG, mode & 0x3)

def fifo_drain():
    regwrite(FIFO_DRAIN_REG, 0x1)

# ============================================================================
# BRAM helpers (indirect: set addr, then read/write)
# ============================================================================

def bram_read(bram_addr):
    regwrite(BRAM_ADDR_REG, bram_addr & 0xFF)
    regread(BRAM_ADDR_REG)   # bus turnaround
    lo   = regread(BRAM_RD_LO_REG)
    hi   = regread(BRAM_RD_HI_REG)
    ctrl = regread(BRAM_RCTRL_REG) & 0xFF
    return (ctrl, hi, lo)

def bram_write(bram_addr, ctrl, hi, lo):
    regwrite(BRAM_ADDR_REG,  bram_addr & 0xFF)
    regwrite(BRAM_WD_LO_REG, lo)
    regwrite(BRAM_WD_HI_REG, hi)
    regwrite(BRAM_WCTRL_REG, ctrl & 0xFF)
    regwrite(BRAM_CMD_REG,   0x1)

# ============================================================================
# CPU helpers
# ============================================================================

def cpu_assert_reset():
    regwrite(CPU_CTRL_REG, 0x2)   # bit[1]=reset, bit[0]=start_stop=0

def cpu_release_reset():
    regwrite(CPU_CTRL_REG, 0x0)

def cpu_start():
    regwrite(CPU_CTRL_REG, 0x1)   # bit[0]=start_stop=1, reset released

def cpu_stop():
    regwrite(CPU_CTRL_REG, 0x0)

def cpu_write_imem(addr, data):
    regwrite(CPU_IMEM_ADDR_REG,  addr & 0x1FF)
    regwrite(CPU_IMEM_WDATA_REG, data)
    regwrite(CPU_IMEM_CMD_REG,   0x1)

def cpu_read_imem(addr):
    regwrite(CPU_IMEM_ADDR_REG, addr & 0x1FF)
    regread(CPU_IMEM_ADDR_REG)
    return regread(CPU_IMEM_RDATA_REG)

def cpu_write_dmem(addr, hi, lo):
    regwrite(CPU_DMEM_ADDR_REG,  addr & 0xFF)
    regwrite(CPU_DMEM_WD_LO_REG, lo)
    regwrite(CPU_DMEM_WD_HI_REG, hi)
    regwrite(CPU_DMEM_CMD_REG,   0x1)

def cpu_read_dmem(addr):
    regwrite(CPU_DMEM_ADDR_REG, addr & 0xFF)
    regread(CPU_DMEM_ADDR_REG)
    lo = regread(CPU_DMEM_RD_LO_REG)
    hi = regread(CPU_DMEM_RD_HI_REG)
    return (hi, lo)

def cpu_read_la(addr):
    regwrite(CPU_LA_ADDR_REG, addr & 0x7FF)
    regread(CPU_LA_ADDR_REG)
    lo = regread(CPU_LA_RD_LO_REG)
    hi = regread(CPU_LA_RD_HI_REG)
    return (hi, lo)

# ============================================================================
# GPU helpers (indirect register access, unlike Lab7's direct memory-map)
# ============================================================================

def gpu_assert_reset():
    regwrite(GPU_CTRL_REG, 0x1)

def gpu_release_reset():
    regwrite(GPU_CTRL_REG, 0x0)

def gpu_read_status():
    return regread(GPU_STATUS_REG)

def gpu_write_imem(addr, data):
    regwrite(GPU_IMEM_ADDR_REG,  addr & 0x3FF)
    regwrite(GPU_IMEM_WDATA_REG, data)
    regwrite(GPU_IMEM_CMD_REG,   0x1)

def gpu_read_imem(addr):
    regwrite(GPU_IMEM_ADDR_REG, addr & 0x3FF)
    regread(GPU_IMEM_ADDR_REG)
    return regread(GPU_IMEM_RDATA_REG)

def gpu_write_dmem(addr, hi, lo):
    regwrite(GPU_DMEM_ADDR_REG,  addr & 0x3FF)
    regwrite(GPU_DMEM_WD_LO_REG, lo)
    regwrite(GPU_DMEM_WD_HI_REG, hi)
    regwrite(GPU_DMEM_CMD_REG,   0x1)

def gpu_read_dmem(addr):
    regwrite(GPU_DMEM_ADDR_REG, addr & 0x3FF)
    regread(GPU_DMEM_ADDR_REG)
    lo = regread(GPU_DMEM_RD_LO_REG)
    hi = regread(GPU_DMEM_RD_HI_REG)
    return (hi, lo)

def gpu_poll_done(timeout_ms=2000):
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        if gpu_read_status() & 1:
            return True
        time.sleep(0.01)
    return False

def gpu_read_cycle_count():
    """Read the hardware cycle counter (counts core_clk cycles from gpu_reset
    deassertion until kernel_done asserted). Returns 0 while the GPU is held
    in reset; frozen after kernel_done; cleared on the next reset release."""
    return regread(GPU_CYCLE_COUNT_REG)

# ============================================================================
# DMA helpers
# ============================================================================

def dma_transfer_and_wait(fifo_addr, gpu_addr, length, direction):
    """direction: 0=FIFO->GPU, 1=GPU->FIFO"""
    regwrite(DMA_FIFO_ADDR_REG, fifo_addr & 0xFF)
    regwrite(DMA_GPU_ADDR_REG,  gpu_addr & 0x3FF)
    regwrite(DMA_LENGTH_REG,    length & 0xFF)
    regwrite(DMA_CTRL_REG, ((direction & 1) << 1) | 1)

    deadline = time.time() + 1.0
    while time.time() < deadline:
        s = regread(GPU_STATUS_REG)
        if (s & 0x2) == 0:
            return True
        time.sleep(0.01)
    print("WARNING: DMA transfer did not complete within 1s")
    return False

# ============================================================================
# Hex file loader
# ============================================================================

def find_hex(hex_file):
    """Search for hex file in common locations."""
    candidates = [
        hex_file,
        os.path.join(".", hex_file),
        os.path.join(SCRIPT_DIR, hex_file),
        os.path.join(SCRIPT_DIR, "..", "programs", hex_file),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return hex_file

def parse_hex_file(hex_path):
    """Parse a hex file, return list of integer words."""
    f = open(hex_path, "r")
    try:
        lines = f.readlines()
    finally:
        f.close()

    words = []
    for line in lines:
        line = re.sub(r"//.*", "", line)
        line = re.sub(r"#.*", "", line)
        line = line.strip()
        if line and re.match(r"^[0-9a-fA-F]+$", line):
            words.append(int(line, 16))
    return words

def load_hex(hex_path):
    """Load hex file into GPU IMEM (asserts GPU reset first)."""
    words = parse_hex_file(hex_path)
    gpu_assert_reset()
    for i in range(len(words)):
        gpu_write_imem(i, words[i])
    print("Loaded %d instructions from %s" % (len(words), os.path.basename(hex_path)))
    return len(words)

def load_dmem_hex(hex_path):
    """Load 64-bit hex file into GPU DMEM. GPU must already be in reset."""
    words = parse_hex_file(hex_path)
    for i in range(len(words)):
        hi = (words[i] >> 32) & 0xFFFFFFFF
        lo = words[i] & 0xFFFFFFFF
        gpu_write_dmem(i, hi, lo)
    print("Loaded %d DMEM words from %s" % (len(words), os.path.basename(hex_path)))
    return len(words)

def load_cpu_hex(hex_path):
    """Load hex file into CPU IMEM (asserts CPU reset first)."""
    f = open(hex_path, "r")
    try:
        lines = f.readlines()
    finally:
        f.close()

    cpu_assert_reset()
    addr = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        cpu_write_imem(addr, int(line, 16))
        addr += 1
    print("Loaded %d instructions from %s (CPU held in reset)" %
          (addr, os.path.basename(hex_path)))
    return addr

# ============================================================================
# Commands -- FIFO
# ============================================================================

def cmd_reset():
    fifo_reset()
    print("FIFO soft reset asserted and released.")

def cmd_status():
    s = regread(STATUS_REG)
    pkt_ready    = s & 1
    fifo_full    = (s >> 1) & 1
    fifo_empty   = (s >> 2) & 1
    tail         = (s >> 8) & 0xFF
    head         = (s >> 16) & 0xFF
    pkt_word_cnt = (s >> 24) & 0xFF

    mode = regread(FIFO_MODE_REG) & 0x3
    mode_names = ["FIFO", "CPU", "GPU", "RESERVED"]

    print("STATUS = 0x%08x" % s)
    print("  pkt_ready=%d  fifo_full=%d  fifo_empty=%d" %
          (pkt_ready, fifo_full, fifo_empty))
    print("  pkt_word_cnt=%d  head=%d  tail=%d" %
          (pkt_word_cnt, head, tail))
    print("  mode=%d (%s)" % (mode, mode_names[mode]))

def cmd_mode(mode_str):
    mode = int(mode_str)
    mode_names = ["FIFO", "CPU", "GPU", "RESERVED"]
    fifo_set_mode(mode)
    print("Mode set to %d (%s)" % (mode, mode_names[mode]))

def cmd_drain():
    fifo_drain()
    print("Drain triggered (auto-clear strobe).")

def cmd_fifo_read(start=0, count=8):
    fifo_set_mode(1)
    for i in range(start, start + count):
        ctrl, hi, lo = bram_read(i)
        print("BRAM[%3d]: ctrl=0x%02x  data=0x%08x_%08x" % (i, ctrl, hi, lo))
    fifo_set_mode(0)

def cmd_fifo_write(addr, hi_hex, lo_hex):
    hi = int(hi_hex, 16)
    lo = int(lo_hex, 16)
    fifo_set_mode(1)
    bram_write(addr, 0x00, hi, lo)
    print("BRAM[%3d] <= ctrl=0x00  data=0x%08x_%08x" % (addr, hi, lo))
    rctrl, rhi, rlo = bram_read(addr)
    print("Readback:   ctrl=0x%02x  data=0x%08x_%08x" % (rctrl, rhi, rlo))
    fifo_set_mode(0)

def cmd_bram_write_full(addr, ctrl_hex, hi_hex, lo_hex):
    ctrl = int(ctrl_hex, 16)
    hi = int(hi_hex, 16)
    lo = int(lo_hex, 16)
    fifo_set_mode(1)
    bram_write(addr, ctrl & 0xFF, hi, lo)
    print("BRAM[%3d] <= ctrl=0x%02x  data=0x%08x_%08x" % (addr, ctrl, hi, lo))
    rctrl, rhi, rlo = bram_read(addr)
    print("Readback:   ctrl=0x%02x  data=0x%08x_%08x" % (rctrl, rhi, rlo))
    fifo_set_mode(0)

def cmd_packet_echo(dst_port_hex=None):
    print("=== Packet Echo Test ===\n")

    # Step 1: Check for packet
    print("1. Checking for packet...")
    s = regread(STATUS_REG)
    pkt_ready = s & 1
    pkt_word_cnt = (s >> 24) & 0xFF
    if not pkt_ready:
        print("   No packet in FIFO. Send a packet from a sub-node first.")
        return

    print("  Packet ready: %d words" % pkt_word_cnt)

    # Step 2: Read header
    print("2. Reading NF2.1 module header (BRAM[0])...")
    fifo_set_mode(1)
    hctrl, hhi, hlo = bram_read(0)
    orig_dst = (hhi >> 16) & 0xFFFF
    word_len = hhi & 0xFFFF
    src_port = (hlo >> 16) & 0xFFFF
    byte_len = hlo & 0xFFFF
    print("   ctrl=0x%02x  dst_port=0x%04x  word_len=%d  src_port=0x%04x  byte_len=%d" %
          (hctrl, orig_dst, word_len, src_port, byte_len))
    print("   src = %s,  dst = %s" %
          (PORT_NAMES.get(src_port, "0x%04x" % src_port),
           PORT_NAMES.get(orig_dst, "0x%04x" % orig_dst)))

    # Step 3: Show data words
    print("3. Packet data words:")
    show = min(pkt_word_cnt, 8)
    for i in range(1, show):
        rc, rh, rl = bram_read(i)
        print("  BRAM[%d]: ctrl=0x%02x  data=0x%08x_%08x" % (i, rc, rh, rl))
    if pkt_word_cnt > 8:
        print("  ... (%d more words)" % (pkt_word_cnt - 8))

    # Step 4: Determine dst_port
    if dst_port_hex is not None:
        new_dst = int(dst_port_hex, 16)
        print("4. Using specified dst_port=0x%04x (%s)" %
              (new_dst, PORT_NAMES.get(new_dst, "unknown")))
    else:
        if src_port == 0:
            print("4. src_port=0x0000 (not set). Must specify dst_port.")
            print("   Usage: lab8reg.py packet_echo <dst_port_hex>")
            print("   Ports: 01=MAC0 04=MAC1 10=MAC2 40=MAC3")
            fifo_set_mode(0)
            return
        new_dst = src_port
        print("4. No dst_port specified -- echoing back to src_port=0x%04x (%s)" %
              (new_dst, PORT_NAMES.get(new_dst, "unknown")))

    # Step 5: Rewrite header
    new_hi = ((new_dst & 0xFFFF) << 16) | (word_len & 0xFFFF)
    print("5. Rewriting header: dst_port 0x%04x -> 0x%04x (ctrl=0xFF preserved)..." %
          (orig_dst, new_dst))
    bram_write(0, 0xFF, new_hi, hlo)
    vc, vh, vl = bram_read(0)
    print("   Verify: ctrl=0x%02x  data=0x%08x_%08x" % (vc, vh, vl))

    # Step 6: Drain
    print("6. Switching to FIFO mode and draining...")
    fifo_set_mode(0)
    fifo_drain()
    print("   Drain triggered.")

    print("\n=== Packet Echo Complete ===")
    print("Packet sent to %s." % PORT_NAMES.get(new_dst, "port 0x%04x" % new_dst))

# ============================================================================
# Commands -- CPU
# ============================================================================

def cmd_cpu_status():
    s = regread(CPU_STATUS_REG)
    not_reset  = s & 1
    running    = (s >> 1) & 1
    thread_id  = (s >> 10) & 0x3
    cpsr_0     = (s >> 12) & 0xF
    cpsr_1     = (s >> 16) & 0xF
    cpsr_2     = (s >> 20) & 0xF
    cpsr_3     = (s >> 24) & 0xF
    all_halted = (s >> 31) & 1

    ctrl = regread(CPU_CTRL_REG)

    print("CPU_STATUS = 0x%08x  CPU_CTRL = 0x%08x" % (s, ctrl))
    print("  all_halted=%d  running=%d  reset=%d  thread_id=%d" %
          (all_halted, running, not not_reset, thread_id))
    print("  cpsr: T0=0x%x  T1=0x%x  T2=0x%x  T3=0x%x" %
          (cpsr_0, cpsr_1, cpsr_2, cpsr_3))

def cmd_cpu_reset():
    cpu_assert_reset()
    print("CPU reset (held in reset).")

def cmd_cpu_start():
    fifo_set_mode(1)
    cpu_start()
    print("CPU started (mode=CPU).")

def cmd_cpu_stop():
    cpu_assert_reset()
    print("CPU stopped (held in reset).")

def cmd_cpu_load_imem(filename):
    load_cpu_hex(find_hex(filename))

def cmd_cpu_read_imem(start=0, count=16):
    for i in range(start, start + count):
        data = cpu_read_imem(i)
        print("IMEM[%3d]: 0x%08x" % (i, data))

def cmd_cpu_read_dmem(start=0, count=8):
    for i in range(start, start + count):
        hi, lo = cpu_read_dmem(i)
        print("DMEM[%3d]: 0x%08x_%08x" % (i, hi, lo))

def cmd_cpu_write_dmem(addr, hi_hex, lo_hex):
    hi = int(hi_hex, 16)
    lo = int(lo_hex, 16)
    cpu_write_dmem(addr, hi, lo)
    print("DMEM[%3d] <= 0x%08x_%08x" % (addr, hi, lo))
    rhi, rlo = cpu_read_dmem(addr)
    print("Readback:    0x%08x_%08x" % (rhi, rlo))

def cmd_cpu_la_read(start=0, count=16):
    for i in range(start, start + count):
        hi, lo = cpu_read_la(i)
        print("LA[%4d]: 0x%08x_%08x" % (i, hi, lo))

def cmd_cpu_test(filename):
    print("=== CPU-FIFO Integration Test ===\n")

    # Step 1: Reset
    print("1. Reset FIFO + CPU...")
    fifo_reset()
    cpu_assert_reset()

    # Step 2: Load IMEM
    print("2. Loading IMEM from %s..." % filename)
    cmd_cpu_load_imem(filename)

    # Step 3: Write BRAM[1] = 0x1000
    print("3. Writing BRAM[1] = 0x00000000_00001000...")
    fifo_set_mode(1)
    bram_write(1, 0x00, 0x00000000, 0x00001000)

    rctrl, rhi, rlo = bram_read(1)
    print("   BRAM[1] verify: ctrl=0x%02x data=0x%08x_%08x" % (rctrl, rhi, rlo))
    if rlo != 0x00001000 or rhi != 0x00000000:
        print("   WARNING: BRAM[1] doesn't match! Re-writing...")
        bram_write(1, 0x00, 0x00000000, 0x00001000)
        rctrl, rhi, rlo = bram_read(1)
        print("   BRAM[1] re-verify: ctrl=0x%02x data=0x%08x_%08x" %
              (rctrl, rhi, rlo))

    # Step 4: Start CPU
    print("4. Starting CPU (mode already CPU)...")
    cpu_start()
    print("   CPU started.")

    # Step 5: Wait for all_halted
    print("5. Waiting for all_halted...")
    halted = False
    for i in range(100):
        s = regread(CPU_STATUS_REG)
        if s & 0x80000000:
            print("   all_halted=1 (CPU_STATUS=0x%08x)" % s)
            halted = True
            break
        time.sleep(0.01)
    if not halted:
        s = regread(CPU_STATUS_REG)
        print("   WARNING: all_halted not set after 1s (CPU_STATUS=0x%08x)" % s)
        print("   Continuing anyway...")

    # Step 6: Stop CPU
    cpu_assert_reset()

    # Step 7: Verify BRAM[1]
    print("6. Reading BRAM[1]...")
    rctrl, rhi, rlo = bram_read(1)
    print("   BRAM[1]: ctrl=0x%02x data=0x%08x_%08x" % (rctrl, rhi, rlo))

    expected_lo = 0x0000102A
    expected_hi = 0x00000000
    if rlo == expected_lo and rhi == expected_hi:
        print("   PASS: BRAM[1] = 0x%08x_%08x (0x1000 + 42 = 0x102A)" %
              (rhi, rlo))
    else:
        print("   FAIL: expected 0x%08x_%08x, got 0x%08x_%08x" %
              (expected_hi, expected_lo, rhi, rlo))
        if rlo == 0x00001000:
            print("   (Value unchanged -- CPU may not have accessed FIFO BRAM)")

    # Step 8: LA trace
    print("\n7. LA Trace (first 10 entries):")
    for i in range(10):
        hi, lo = cpu_read_la(i)
        print("   LA[%2d]: 0x%08x_%08x" % (i, hi, lo))

    fifo_set_mode(0)
    print("\nDone. Mode set back to FIFO.")

# ============================================================================
# Commands -- GPU
# ============================================================================

def cmd_gpu_status():
    s = regread(GPU_STATUS_REG)
    kernel_done = s & 1
    dma_busy    = (s >> 1) & 1
    ctrl = regread(GPU_CTRL_REG)
    gpu_reset = ctrl & 1

    print("GPU_STATUS = 0x%08x  GPU_CTRL = 0x%08x" % (s, ctrl))
    print("  kernel_done=%d  dma_busy=%d  gpu_reset=%d" %
          (kernel_done, dma_busy, gpu_reset))

def cmd_gpu_reset():
    gpu_assert_reset()
    print("GPU held in reset.")

def cmd_gpu_start():
    gpu_release_reset()
    print("GPU released from reset (executing).")

def cmd_gpu_stop():
    gpu_assert_reset()
    print("GPU stopped (held in reset).")

def cmd_gpu_load_imem(filename):
    load_hex(find_hex(filename))

def cmd_gpu_read_imem(start=0, count=16):
    for i in range(start, start + count):
        data = gpu_read_imem(i)
        print("GPU IMEM[%3d]: 0x%08x" % (i, data))

def cmd_gpu_read_dmem(start=0, count=8):
    for i in range(start, start + count):
        hi, lo = gpu_read_dmem(i)
        print("GPU DMEM[%3d]: 0x%08x_%08x" % (i, hi, lo))

def cmd_gpu_write_dmem(addr, hi_hex, lo_hex):
    hi = int(hi_hex, 16)
    lo = int(lo_hex, 16)
    gpu_write_dmem(addr, hi, lo)
    print("GPU DMEM[%3d] <= 0x%08x_%08x" % (addr, hi, lo))
    rhi, rlo = gpu_read_dmem(addr)
    print("Readback:       0x%08x_%08x" % (rhi, rlo))

# ============================================================================
# Commands -- DMA
# ============================================================================

def cmd_dma_fifo_to_gpu(fifo_addr, gpu_addr, length):
    print("DMA FIFO->GPU: FIFO[%d..%d] -> GPU DMEM[%d..%d] (%d words)" %
          (fifo_addr, fifo_addr + length - 1,
           gpu_addr,  gpu_addr + length - 1, length))
    if dma_transfer_and_wait(fifo_addr, gpu_addr, length, 0):
        print("DMA complete.")

def cmd_dma_gpu_to_fifo(gpu_addr, fifo_addr, length):
    print("DMA GPU->FIFO: GPU DMEM[%d..%d] -> FIFO[%d..%d] (%d words)" %
          (gpu_addr,  gpu_addr + length - 1,
           fifo_addr, fifo_addr + length - 1, length))
    if dma_transfer_and_wait(fifo_addr, gpu_addr, length, 1):
        print("DMA complete.")

# ============================================================================
# Commands -- Integrated GPU+DMA test (with captured packet)
# ============================================================================

def cmd_gpu_test(filename):
    print("=== GPU-DMA Integration Test ===\n")

    # Step 1: Reset GPU (keep FIFO packet intact)
    print("1. Reset GPU...")
    gpu_assert_reset()

    # Step 2: Check for packet
    print("2. Checking for packet...")
    s = regread(STATUS_REG)
    pkt_ready = s & 1
    pkt_word_cnt = (s >> 24) & 0xFF
    if not pkt_ready:
        print("   No packet in FIFO. Send a packet first, then retry.")
        return

    print("   Packet ready: %d words" % pkt_word_cnt)

    # Step 3: GPU mode
    print("3. Setting mode=GPU...")
    fifo_set_mode(2)

    # Step 4: Load GPU IMEM
    print("4. Loading GPU IMEM from %s..." % filename)
    load_hex(find_hex(filename))

    # Step 5: DMA FIFO->GPU (skip header, copy data words 1..N)
    data_words = pkt_word_cnt - 1
    print("5. DMA FIFO->GPU: FIFO[1..%d] -> GPU DMEM[0..%d] (%d words)..." %
          (pkt_word_cnt - 1, data_words - 1, data_words))
    if not dma_transfer_and_wait(1, 0, data_words, 0):
        print("   DMA failed!")
        return
    print("   DMA complete.")

    # Step 6: Verify GPU DMEM[0]
    print("6. Verifying GPU DMEM[0]...")
    hi, lo = gpu_read_dmem(0)
    print("   GPU DMEM[0] = 0x%08x_%08x" % (hi, lo))
    rctrl, rhi, rlo = bram_read(1)
    print("   FIFO BRAM[1] = 0x%08x_%08x (ctrl=0x%02x)" % (rhi, rlo, rctrl))

    # Step 7: Run kernel
    print("7. Running GPU kernel...")
    gpu_release_reset()
    done = False
    for i in range(200):
        s = regread(GPU_STATUS_REG)
        if s & 1:
            print("   kernel_done after polling %d times" % i)
            done = True
            break
        time.sleep(0.01)
    if not done:
        print("   WARNING: kernel_done not set after 2s (GPU_STATUS=0x%08x)" %
              regread(GPU_STATUS_REG))
    gpu_assert_reset()

    # Step 8: Show GPU DMEM
    print("8. GPU DMEM after kernel:")
    for i in range(min(data_words, 8)):
        hi, lo = gpu_read_dmem(i)
        print("   DMEM[%d] = 0x%08x_%08x" % (i, hi, lo))

    # Step 9: DMA GPU->FIFO
    print("9. DMA GPU->FIFO: GPU DMEM[0..%d] -> FIFO[1..%d]..." %
          (data_words - 1, pkt_word_cnt - 1))
    if not dma_transfer_and_wait(1, 0, data_words, 1):
        print("   DMA failed!")
        return
    print("   DMA complete.")

    # Step 10: Verify FIFO BRAM
    print("10. FIFO BRAM after round-trip:")
    for i in range(min(pkt_word_cnt, 9)):
        rctrl, rhi, rlo = bram_read(i)
        print("    BRAM[%d]: ctrl=0x%02x data=0x%08x_%08x" % (i, rctrl, rhi, rlo))

    # Step 11: Drain
    print("11. Switching to FIFO mode and draining...")
    fifo_set_mode(0)
    fifo_drain()
    print("    Drain triggered.")

    print("\n=== GPU-DMA Test Complete ===")

# ============================================================================
# GPU standalone test suite (PCI-direct, no DMA, no packet required)
# ============================================================================

def run_and_check(test_name, hex_file, dmem_data, result_addr,
                  expect_hi, expect_lo):
    """Load kernel, write test data to GPU DMEM via PCI, run, check result."""
    print("=== %s ===" % test_name)

    load_hex(find_hex(hex_file))

    for addr, hi, lo in dmem_data:
        gpu_write_dmem(addr, hi, lo)

    # Debug: verify PCI writes before running kernel
    write_ok = True
    for addr, hi, lo in dmem_data:
        rhi, rlo = gpu_read_dmem(addr)
        if rhi != hi or rlo != lo:
            print("  READBACK MISMATCH: DMEM[%d] wrote 0x%08x_%08x, read 0x%08x_%08x" %
                  (addr, hi, lo, rhi, rlo))
            write_ok = False
    if write_ok:
        print("  PCI DMEM writes verified OK (%d words)" % len(dmem_data))
    else:
        print("  WARNING: PCI DMEM writes have mismatches!")

    # Also read the result address before kernel (to see stale value)
    pre_hi, pre_lo = gpu_read_dmem(result_addr)
    print("  DMEM[%d] before kernel: 0x%08x_%08x" % (result_addr, pre_hi, pre_lo))

    gpu_release_reset()
    if not gpu_poll_done():
        print("  TIMEOUT: kernel did not complete")
        print("")
        return False

    print("  kernel_done=1")

    # Re-assert reset before reading (stop GPU from executing further)
    gpu_assert_reset()

    rhi, rlo = gpu_read_dmem(result_addr)
    print("  DMEM[0x%03x] = 0x%08x_%08x" % (result_addr, rhi, rlo))
    print("  Expected:    0x%08x_%08x" % (expect_hi, expect_lo))

    if rhi == expect_hi and rlo == expect_lo:
        print("  !!! PASS !!!")
        print("")
        return True
    else:
        print("  !!! FAIL !!!")
        # Dump all DMEM used by this test for debugging
        max_addr = result_addr
        for a, h, l in dmem_data:
            if a > max_addr:
                max_addr = a
        print("  DMEM dump after kernel:")
        for i in range(max_addr + 1):
            dh, dl = gpu_read_dmem(i)
            print("    DMEM[%d] = 0x%08x_%08x" % (i, dh, dl))
        print("")
        return False

def run_and_check_files(test_name, imem_hex, dmem_hex, result_addr,
                        expect_hi, expect_lo):
    """Test with IMEM and DMEM loaded from separate hex files."""
    print("=== %s ===" % test_name)
    load_hex(find_hex(imem_hex))
    load_dmem_hex(find_hex(dmem_hex))

    gpu_release_reset()
    if not gpu_poll_done():
        print("  TIMEOUT: kernel did not complete")
        print("")
        return False

    print("  kernel_done=1")
    gpu_assert_reset()

    rhi, rlo = gpu_read_dmem(result_addr)
    print("  DMEM[0x%03x] = 0x%08x_%08x" % (result_addr, rhi, rlo))
    print("  Expected:    0x%08x_%08x" % (expect_hi, expect_lo))

    if rhi == expect_hi and rlo == expect_lo:
        print("  !!! PASS !!!")
        print("")
        return True
    else:
        print("  !!! FAIL !!!")
        print("")
        return False

# --- 5 CUDA kernel tests ---

def test_debug():
    """Verbose diagnostic for vec_add: verify IMEM, DMEM, run, dump everything."""
    print("=== DEBUG: vec_add_i16 full trace ===\n")

    hex_path = find_hex("vec_add_i16.hex")
    print("1. Loading IMEM from %s..." % hex_path)
    load_hex(hex_path)

    # Expected IMEM contents for vec_add_i16:
    # 0: LD R1, 0(R0)     1: LD R2, 1(R0)     2: ADD R3, R1, R2
    # 3: ST R3, 2(R0)     4: HALT
    print("\n2. IMEM readback (addresses 0-7):")
    for i in range(8):
        val = gpu_read_imem(i)
        print("   IMEM[%d] = 0x%08x" % (i, val))

    print("\n3. Writing DMEM test data...")
    gpu_write_dmem(0, 0x000A0014, 0x001E0028)
    gpu_write_dmem(1, 0x00010002, 0x00030004)

    print("\n4. DMEM readback before kernel (addresses 0-4):")
    for i in range(5):
        hi, lo = gpu_read_dmem(i)
        print("   DMEM[%d] = 0x%08x_%08x" % (i, hi, lo))

    print("\n5. Releasing GPU from reset...")
    gpu_release_reset()
    if not gpu_poll_done(3000):
        print("   TIMEOUT!")
        gpu_assert_reset()
        return

    print("   kernel_done=1")
    gpu_assert_reset()

    print("\n6. DMEM after kernel (addresses 0-7):")
    for i in range(8):
        hi, lo = gpu_read_dmem(i)
        tag = ""
        if i == 2:
            tag = "  <-- result (expected 0x000b0016_0021002c)"
        print("   DMEM[%d] = 0x%08x_%08x%s" % (i, hi, lo, tag))

    print("\n7. GPU status:")
    s = regread(GPU_STATUS_REG)
    print("   GPU_STATUS = 0x%08x" % s)

    print("\n=== DEBUG complete ===")

def test_vec_add():
    return run_and_check("vec_add_i16: R3 = R1 + R2", "vec_add_i16.hex",
        [(0, 0x000A0014, 0x001E0028), (1, 0x00010002, 0x00030004)],
        2, 0x000B0016, 0x0021002C)

def test_vec_sub():
    return run_and_check("vec_sub_i16: R3 = R1 - R2", "vec_sub_i16.hex",
        [(0, 0x000A0014, 0x001E0028), (1, 0x00030005, 0x000A000F)],
        2, 0x0007000F, 0x00140019)

def test_bf16_mul():
    return run_and_check("bf16_mul: R3 = R1 * R2 (BF16)", "bf16_mul.hex",
        [(0, 0x3F804000, 0x3F004040), (1, 0x40004040, 0x3F003E80)],
        2, 0x400040C0, 0x3E803F40)

def test_bf16_fma():
    return run_and_check("bf16_fma: R4 = R1*R2 + R3 (BF16)", "bf16_fma.hex",
        [(0, 0x3F804000, 0x3F004040), (1, 0x40004040, 0x3F004000),
         (2, 0x40403F80, 0x3E803F00)],
        3, 0x40A040E0, 0x3F0040D0)

def test_relu():
    return run_and_check("relu_i16: R2 = max(0, R1)", "relu_i16.hex",
        [(0, 0xFFFD0005, 0xFFF90007)],
        1, 0x00000005, 0x00000007)

# --- 2 ANN inference tests ---

def test_ann():
    return run_and_check_files(
        "ANN handcrafted: f(x)=3*ReLU(0.5*ReLU(2x-1)+0.25)-0.5",
        "ann_3layer_relu.hex", "data_ann_handcrafted.hex",
        7, 0x3E803E80, 0x405040F8)

def test_ann_trained():
    return run_and_check_files(
        "ANN trained: f(x)=clamp(2x-1,0,3)",
        "ann_3layer_relu.hex", "data_ann_trained.hex",
        7, 0x3C413F28, 0x3FCA403C)

# ============================================================================
# Integrated ANN tests (synthetic packet -> DMA -> GPU -> DMA -> verify)
#
# These test the full Lab8 network processor data path:
#   PCI writes a synthetic packet into FIFO BRAM (simulating network input)
#   -> DMA copies payload to GPU DMEM -> GPU runs ANN kernel
#   -> DMA copies results back to FIFO BRAM -> PCI reads back to verify
# ============================================================================

def _build_ann_packet(dmem_data, result_addr):
    """Build a synthetic NF2.1 packet in FIFO BRAM containing ANN data.

    Returns the number of data words written (excluding header).
    Packet is sized to hold result_addr+1 data words (so DMA back includes result).
    Packet layout:
      BRAM[0]: header  ctrl=0xFF
      BRAM[1..N]: data  ctrl=0x00, 64-bit payload
    """
    total_data = result_addr + 1   # enough room for result
    word_len = total_data + 1      # header + data
    byte_len = total_data * 8      # 8 bytes per data word

    # NF2.1 header: {dst_port[15:0], word_length[15:0], src_port[15:0], byte_length[15:0]}
    hdr_hi = (0x0001 << 16) | (word_len & 0xFFFF)   # dst=MAC0, word_len
    hdr_lo = (0x0002 << 16) | (byte_len & 0xFFFF)   # src=CPU0, byte_len

    fifo_reset()
    fifo_set_mode(1)   # CPU mode for PCI BRAM access

    # Write header
    bram_write(0, 0xFF, hdr_hi, hdr_lo)

    # Write data words as packet payload
    for i in range(len(dmem_data)):
        addr, hi, lo = dmem_data[i]
        bram_write(i + 1, 0x00, hi, lo)

    fifo_set_mode(0)   # back to FIFO mode

    return total_data

def _run_integrated_ann(test_name, hex_file, dmem_data, result_addr,
                        expect_hi, expect_lo):
    """Full network-processor ANN test with synthetic packet."""
    print("=== %s (integrated) ===" % test_name)

    # Step 1: Build synthetic packet in FIFO BRAM
    print("  1. Building synthetic packet (%d input words, result at DMEM[%d])..." %
          (len(dmem_data), result_addr))
    data_words = _build_ann_packet(dmem_data, result_addr)

    # Manually set pkt_stored so FIFO thinks a packet arrived.
    # We do this by writing the FIFO status via the ctrl register trick:
    # Actually, the FIFO FSM only sets pkt_stored when it receives via Port A.
    # Since we wrote via Port B (PCI), pkt_stored=0 and pkt_ready=0.
    # For the integrated test, we bypass the FIFO FSM and drive DMA directly.
    # The DMA engine only checks mode==2 and length!=0, not pkt_ready.

    # Step 2: Set GPU mode
    print("  2. Setting mode=GPU...")
    fifo_set_mode(2)

    # Step 3: Load GPU IMEM
    print("  3. Loading GPU IMEM...")
    load_hex(find_hex(hex_file))

    # Step 4: DMA FIFO->GPU (BRAM[1..N] -> GPU DMEM[0..N-1])
    print("  4. DMA FIFO->GPU: BRAM[1..%d] -> DMEM[0..%d]..." %
          (data_words, data_words - 1))
    if not dma_transfer_and_wait(1, 0, data_words, 0):
        print("  DMA FIFO->GPU failed!")
        print("")
        return False
    print("     DMA complete.")

    # Step 5: Verify DMA transfer (spot check DMEM[0])
    hi, lo = gpu_read_dmem(0)
    print("  5. GPU DMEM[0] = 0x%08x_%08x (should match BRAM[1])" % (hi, lo))

    # Step 6: Run GPU kernel
    print("  6. Running GPU kernel...")
    gpu_release_reset()
    if not gpu_poll_done():
        print("  TIMEOUT: kernel did not complete")
        print("")
        return False
    print("     kernel_done=1")
    gpu_assert_reset()

    # Step 7: Read GPU result before DMA back
    rhi, rlo = gpu_read_dmem(result_addr)
    print("  7. GPU DMEM[%d] = 0x%08x_%08x" % (result_addr, rhi, rlo))

    # Step 8: DMA GPU->FIFO -- must include result_addr
    dma_back_len = result_addr + 1
    print("  8. DMA GPU->FIFO: DMEM[0..%d] -> BRAM[1..%d]..." %
          (dma_back_len - 1, dma_back_len))
    if not dma_transfer_and_wait(1, 0, dma_back_len, 1):
        print("  DMA GPU->FIFO failed!")
        print("")
        return False
    print("     DMA complete.")

    # Step 9: Read result from FIFO BRAM (result_addr+1 because BRAM[0]=header)
    fifo_set_mode(1)
    bram_idx = result_addr + 1
    rctrl, bhi, blo = bram_read(bram_idx)
    fifo_set_mode(0)
    print("  9. FIFO BRAM[%d] = 0x%08x_%08x (round-trip result)" %
          (bram_idx, bhi, blo))
    print("     Expected:     0x%08x_%08x" % (expect_hi, expect_lo))

    if bhi == expect_hi and blo == expect_lo:
        print("  !!! PASS !!!")
        print("")
        return True
    else:
        print("  !!! FAIL !!!")
        # Show full BRAM for debugging
        print("  BRAM dump:")
        fifo_set_mode(1)
        for i in range(min(data_words + 1, 9)):
            rc, rh, rl = bram_read(i)
            print("    BRAM[%d]: ctrl=0x%02x data=0x%08x_%08x" % (i, rc, rh, rl))
        fifo_set_mode(0)
        print("")
        return False

def test_ann_integrated():
    """ANN with hand-crafted weights, full FIFO->DMA->GPU->DMA->FIFO path."""
    dmem_words = parse_hex_file(find_hex("data_ann_handcrafted.hex"))
    dmem_data = []
    for i in range(len(dmem_words)):
        hi = (dmem_words[i] >> 32) & 0xFFFFFFFF
        lo = dmem_words[i] & 0xFFFFFFFF
        dmem_data.append((i, hi, lo))
    return _run_integrated_ann(
        "ANN handcrafted via network packet", "ann_3layer_relu.hex",
        dmem_data, 7, 0x3E803E80, 0x405040F8)

def test_ann_trained_integrated():
    """ANN with trained weights, full FIFO->DMA->GPU->DMA->FIFO path."""
    dmem_words = parse_hex_file(find_hex("data_ann_trained.hex"))
    dmem_data = []
    for i in range(len(dmem_words)):
        hi = (dmem_words[i] >> 32) & 0xFFFFFFFF
        lo = dmem_words[i] & 0xFFFFFFFF
        dmem_data.append((i, hi, lo))
    return _run_integrated_ann(
        "ANN trained via network packet", "ann_3layer_relu.hex",
        dmem_data, 7, 0x3C413F28, 0x3FCA403C)

# ============================================================================
# Real network packet ANN tests (hybrid: weights via PCI, input from packet DMA)
#
# These test ANN inference on actual captured network packets (e.g. ARP from
# node0), proving the full data path with real network data.  The test uses
# a PCI-direct reference run to compute expected output, then verifies the
# DMA path produces the same result.
# ============================================================================

def _load_weights_from_hex(hex_path):
    """Load ANN weights from DMEM hex file into GPU DMEM[1-6], clear DMEM[7].
    Skips DMEM[0] (input) -- that comes from the packet via DMA."""
    words = parse_hex_file(hex_path)
    for i in range(1, min(len(words), 7)):
        hi = (words[i] >> 32) & 0xFFFFFFFF
        lo = words[i] & 0xFFFFFFFF
        gpu_write_dmem(i, hi, lo)
    gpu_write_dmem(7, 0, 0)   # clear output slot

def _test_ann_real_packet(test_name, weight_hex):
    """ANN inference on real network packet data.

    Hybrid approach: weights loaded via PCI, input comes from captured packet
    via DMA.  Uses PCI-direct reference run to compute expected output, then
    verifies the DMA path matches.
    """
    print("=== %s (real packet) ===" % test_name)

    # Step 1: Check for captured packet
    print("  1. Checking for captured packet...")
    s = regread(STATUS_REG)
    pkt_ready = s & 1
    pkt_word_cnt = (s >> 24) & 0xFF
    if not pkt_ready:
        print("     No packet in FIFO. Capture one first (reset, wait for ARP).")
        print("  !!! SKIP !!!")
        print("")
        return False
    print("     Packet ready: %d words" % pkt_word_cnt)

    # Step 2: Read packet input word (BRAM[1] = first data word after header)
    print("  2. Reading packet input (BRAM[1])...")
    fifo_set_mode(1)
    hctrl, hhi, hlo = bram_read(0)
    pctrl, pkt_hi, pkt_lo = bram_read(1)
    fifo_set_mode(0)
    print("     Header:  ctrl=0x%02x  data=0x%08x_%08x" % (hctrl, hhi, hlo))
    src_port = (hlo >> 16) & 0xFFFF
    print("     src=%s" % PORT_NAMES.get(src_port, "0x%04x" % src_port))
    print("     BRAM[1]: ctrl=0x%02x  data=0x%08x_%08x  (ANN input)" %
          (pctrl, pkt_hi, pkt_lo))
    print("     Input as 4x BF16: [0x%04x, 0x%04x, 0x%04x, 0x%04x]" %
          ((pkt_hi >> 16) & 0xFFFF, pkt_hi & 0xFFFF,
           (pkt_lo >> 16) & 0xFFFF, pkt_lo & 0xFFFF))

    # Step 3: PCI-direct reference run (compute expected output)
    print("  3. PCI-direct reference run...")
    weight_path = find_hex(weight_hex)
    load_hex(find_hex("ann_3layer_relu.hex"))
    _load_weights_from_hex(weight_path)
    gpu_write_dmem(0, pkt_hi, pkt_lo)
    gpu_release_reset()
    if not gpu_poll_done():
        print("     TIMEOUT on reference run!")
        gpu_assert_reset()
        print("  !!! FAIL !!!")
        print("")
        return False
    gpu_assert_reset()
    ref_hi, ref_lo = gpu_read_dmem(7)
    print("     Reference DMEM[7] = 0x%08x_%08x" % (ref_hi, ref_lo))
    print("     Output as 4x BF16: [0x%04x, 0x%04x, 0x%04x, 0x%04x]" %
          ((ref_hi >> 16) & 0xFFFF, ref_hi & 0xFFFF,
           (ref_lo >> 16) & 0xFFFF, ref_lo & 0xFFFF))

    # Step 4: DMA path -- load kernel + weights, DMA input from packet
    print("  4. Setting mode=GPU, loading kernel + weights...")
    fifo_set_mode(2)
    load_hex(find_hex("ann_3layer_relu.hex"))
    _load_weights_from_hex(weight_path)

    print("  5. DMA FIFO->GPU: BRAM[1] -> DMEM[0] (1 word)...")
    if not dma_transfer_and_wait(1, 0, 1, 0):
        print("     DMA FIFO->GPU failed!")
        fifo_set_mode(0)
        print("  !!! FAIL !!!")
        print("")
        return False
    print("     DMA complete.")

    # Verify DMA transferred correctly
    dma_in_hi, dma_in_lo = gpu_read_dmem(0)
    print("     GPU DMEM[0] = 0x%08x_%08x" % (dma_in_hi, dma_in_lo))
    if dma_in_hi != pkt_hi or dma_in_lo != pkt_lo:
        print("     WARNING: DMA input mismatch vs BRAM[1]!")

    # Step 6: Run kernel
    print("  6. Running GPU kernel...")
    gpu_release_reset()
    if not gpu_poll_done():
        print("     TIMEOUT: kernel did not complete")
        gpu_assert_reset()
        fifo_set_mode(0)
        print("  !!! FAIL !!!")
        print("")
        return False
    print("     kernel_done=1")
    gpu_assert_reset()

    # Step 7: Read DMA-path result
    dma_hi, dma_lo = gpu_read_dmem(7)
    print("  7. DMA-path DMEM[7]  = 0x%08x_%08x" % (dma_hi, dma_lo))

    # Step 8: DMA results back to FIFO BRAM
    print("  8. DMA GPU->FIFO: DMEM[0-7] -> BRAM[1-8]...")
    if not dma_transfer_and_wait(1, 0, 8, 1):
        print("     DMA GPU->FIFO failed!")
        fifo_set_mode(0)
        print("  !!! FAIL !!!")
        print("")
        return False
    print("     DMA complete.")

    # Step 9: Read round-trip result from BRAM
    fifo_set_mode(1)
    rctrl, bram_hi, bram_lo = bram_read(8)   # DMEM[7] -> BRAM[1+7=8]
    fifo_set_mode(0)
    print("  9. FIFO BRAM[8]      = 0x%08x_%08x  (round-trip)" %
          (bram_hi, bram_lo))

    # Step 10: Verification
    print("  10. Verification:")
    pass_ref = (dma_hi == ref_hi and dma_lo == ref_lo)
    pass_rt  = (bram_hi == dma_hi and bram_lo == dma_lo)

    if pass_ref:
        print("      DMA vs PCI-direct: MATCH")
    else:
        print("      DMA vs PCI-direct: MISMATCH")
        print("        PCI-direct: 0x%08x_%08x" % (ref_hi, ref_lo))
        print("        DMA path:   0x%08x_%08x" % (dma_hi, dma_lo))

    if pass_rt:
        print("      BRAM round-trip:   MATCH")
    else:
        print("      BRAM round-trip:   MISMATCH")
        print("        DMEM[7]:  0x%08x_%08x" % (dma_hi, dma_lo))
        print("        BRAM[8]:  0x%08x_%08x" % (bram_hi, bram_lo))

    if pass_ref and pass_rt:
        print("  !!! PASS !!!")
        print("")
        return True
    else:
        print("  !!! FAIL !!!")
        print("")
        return False

def test_ann_real_pkt():
    """ANN with hand-crafted weights on real packet data."""
    return _test_ann_real_packet(
        "ANN handcrafted (real packet input)",
        "data_ann_handcrafted.hex")

def test_ann_trained_real_pkt():
    """ANN with trained weights on real packet data."""
    return _test_ann_real_packet(
        "ANN trained (real packet input)",
        "data_ann_trained.hex")

def test_all_real_pkt():
    """Run both ANN tests on real packet data. Requires captured packet."""
    t = 0
    t += test_ann_real_pkt()
    t += test_ann_trained_real_pkt()
    print("=== Real Packet ANN Tests: %d/2 passed ===" % t)
    return t == 2

# ============================================================================
# Crafted-input ANN demo (send known BF16 input from node0, verify equation)
#
# Two handcrafted networks with pre-calculated expected outputs:
#   Network A: f(x) = 3*ReLU(0.5*ReLU(2x-1)+0.25) - 0.5  (original)
#   Network B: f(x) = 2x + 1  (simple linear, identity ReLU layers)
#
# Input sent from node0 via send_bf16.py (UDP with 6-byte padding).
# BF16 data lands at BRAM[7] (byte offset 48, aligned to 8-byte word).
# Weights loaded via PCI, input DMA'd from BRAM[7] to DMEM[0].
# Does NOT DMA back -- BRAM preserved for running both networks on same packet.
# ============================================================================

# BF16 lookup for common values (for display)
_BF16_NAMES = {
    0x0000: '0', 0x3E80: '0.25', 0x3F00: '0.5', 0x3F80: '1.0',
    0x3FC0: '1.5', 0x3FE0: '1.75', 0x4000: '2.0', 0x4020: '2.5',
    0x4040: '3.0', 0x4060: '3.5', 0x4080: '4.0', 0x4098: '4.75',
    0x40A0: '5.0', 0x40B0: '5.5', 0x40C0: '6.0', 0x40E0: '7.0',
    0x40F8: '7.75', 0x4100: '8.0', 0x4108: '8.5', 0x4110: '9.0',
    0x4120: '10.0', 0x4138: '11.5',
    0x412C: '10.75',
    0x7FC0: 'NaN', 0x8000: '-0',
    0xBF00: '-0.5', 0xBF80: '-1.0',
}

def _bf16_str(val):
    """Convert a 16-bit BF16 value to a display string."""
    val = val & 0xFFFF
    if val in _BF16_NAMES:
        return _BF16_NAMES[val]
    # Generic decode
    sign = (val >> 15) & 1
    exp = (val >> 7) & 0xFF
    mant = val & 0x7F
    if exp == 0xFF:
        if mant != 0:
            return 'NaN'
        if sign:
            return '-Inf'
        return 'Inf'
    if exp == 0:
        return '0'
    f = (1.0 + mant / 128.0) * (2 ** (exp - 127))
    if sign:
        f = -f
    # Show as integer if clean, otherwise 4 sig figs
    if f == int(f) and abs(f) < 100000:
        return '%d' % int(f)
    return '%.4g' % f

def _show_bf16_word(hi, lo):
    """Format a 64-bit word as 4x BF16 with float values."""
    v0 = (hi >> 16) & 0xFFFF
    v1 = hi & 0xFFFF
    v2 = (lo >> 16) & 0xFFFF
    v3 = lo & 0xFFFF
    return '{%s, %s, %s, %s}' % (_bf16_str(v0), _bf16_str(v1),
                                  _bf16_str(v2), _bf16_str(v3))

def _test_ann_from_bram(test_name, equation, weight_hex, bram_src,
                        expect_hi, expect_lo):
    """ANN inference on crafted input from a specific BRAM word.

    Weights loaded via PCI to DMEM[1-6], input DMA'd from BRAM[bram_src]
    to DMEM[0].  Does NOT DMA back (preserves BRAM for subsequent tests).
    """
    print("=== %s ===" % test_name)
    print("  Equation: %s" % equation)

    # Check packet
    s = regread(STATUS_REG)
    if not (s & 1):
        print("  No packet in FIFO. Send one from node0 first.")
        print("  !!! SKIP !!!")
        print("")
        return False

    # Read input from BRAM
    fifo_set_mode(1)
    ctrl, in_hi, in_lo = bram_read(bram_src)
    fifo_set_mode(0)
    print("  Input  BRAM[%d]: 0x%08x_%08x  = %s" %
          (bram_src, in_hi, in_lo, _show_bf16_word(in_hi, in_lo)))

    # Load kernel + weights (GPU mode)
    fifo_set_mode(2)
    load_hex(find_hex("ann_3layer_relu.hex"))
    _load_weights_from_hex(find_hex(weight_hex))

    # DMA input from BRAM
    if not dma_transfer_and_wait(bram_src, 0, 1, 0):
        print("  DMA failed!")
        fifo_set_mode(0)
        print("  !!! FAIL !!!")
        print("")
        return False

    # Run kernel
    gpu_release_reset()
    if not gpu_poll_done():
        print("  TIMEOUT!")
        gpu_assert_reset()
        fifo_set_mode(0)
        print("  !!! FAIL !!!")
        print("")
        return False
    gpu_assert_reset()
    fifo_set_mode(0)

    # Read result
    rhi, rlo = gpu_read_dmem(7)
    print("  Output DMEM[7]: 0x%08x_%08x  = %s" %
          (rhi, rlo, _show_bf16_word(rhi, rlo)))
    print("  Expected:       0x%08x_%08x  = %s" %
          (expect_hi, expect_lo, _show_bf16_word(expect_hi, expect_lo)))

    if rhi == expect_hi and rlo == expect_lo:
        print("  !!! PASS !!!")
        print("")
        return True
    else:
        print("  !!! FAIL !!!")
        print("")
        return False

def test_udp_net_a():
    """Handcrafted ANN f(x) = 3*ReLU(0.5*ReLU(2x-1)+0.25)-0.5 on UDP input.
    Input {1.0,2.0,3.0,4.0} -> {1.75, 4.75, 7.75, ?}
    For x<=0.5: f(x)=0.25. For x>0.5: f(x)=3*(0.5*(2x-1)+0.25)-0.5 = 3x-1.25
    x=1: 1.75, x=2: 4.75, x=3: 7.75, x=4: 10.75"""
    # Pre-calculate: f(1)=1.75(3FE0), f(2)=4.75(4098), f(3)=7.75(40F8), f(4)=10.75(412C)
    return _test_ann_from_bram(
        "Network A: handcrafted (UDP input)",
        "f(x) = 3*ReLU(0.5*ReLU(2x-1)+0.25) - 0.5",
        "data_ann_handcrafted.hex", 7,
        0x3FE04098, 0x40F8412C)

def test_udp_net_b():
    """Simple linear ANN f(x) = 2x + 1 on UDP input.
    Input {1.0, 2.0, 3.0, 4.0} -> {3.0, 5.0, 7.0, 9.0}"""
    return _test_ann_from_bram(
        "Network B: linear (UDP input)",
        "f(x) = 2x + 1",
        "data_net_2x_plus_1.hex", 7,
        0x404040A0, 0x40E04110)

def test_udp_both():
    """Run both handcrafted networks on same UDP packet. No reset between."""
    t = 0
    t += test_udp_net_a()
    t += test_udp_net_b()
    print("=== Crafted-Input ANN Demo: %d/2 passed ===" % t)
    return t == 2

def test_all_integrated():
    """Run all 9 tests: 7 standalone + 2 integrated ANN."""
    t = 0
    print("--- Standalone GPU Tests (PCI-direct) ---\n")
    t += test_vec_add()
    t += test_vec_sub()
    t += test_bf16_mul()
    t += test_bf16_fma()
    t += test_relu()
    t += test_ann()
    t += test_ann_trained()
    print("--- Integrated Network Processor Tests (FIFO->DMA->GPU->DMA->FIFO) ---\n")
    t += test_ann_integrated()
    t += test_ann_trained_integrated()
    print("=== TOTAL: %d/9 passed ===" % t)
    return t == 9

# ============================================================================
# Main -- command dispatcher
# ============================================================================

def usage():
    print("""Usage: python lab8reg.py <command> [args...]

  FIFO Control:
    reset                           Soft reset FIFO
    status                          Show FIFO status (pkt_ready, full, empty)
    mode <0|1|2>                    Set FIFO mode (0=FIFO, 1=CPU, 2=GPU)
    drain                           Trigger packet drain

  BRAM Access (auto-switches to CPU mode and back):
    fifo_read [start] [count]       Read BRAM words (default: 0, 8)
    fifo_write <addr> <hi> <lo>     Write 64-bit BRAM word (hex, ctrl=0x00)
    bram_write_full <a> <c> <h> <l> Write BRAM with explicit ctrl byte (hex)

  Packet Routing:
    packet_echo [dst_port_hex]      Inspect header -> rewrite dst_port -> drain
                                    Ports: 01=MAC0 04=MAC1 10=MAC2 40=MAC3

  CPU Control:
    cpu_status                      Show CPU status (halted, running, cpsr)
    cpu_reset                       Assert CPU reset (held in reset)
    cpu_start                       Start CPU (also sets mode=CPU)
    cpu_stop                        Stop CPU (held in reset)

  CPU Memory:
    cpu_load_imem <hex_file>        Load hex file into CPU IMEM
    cpu_read_imem [start] [count]   Read CPU IMEM (default: 0, 16)
    cpu_read_dmem [start] [count]   Read CPU DMEM (default: 0, 8)
    cpu_write_dmem <addr> <hi> <lo> Write 64-bit CPU DMEM word (hex)

  CPU Debug:
    cpu_la_read [start] [count]     Read logic analyzer (default: 0, 16)
    cpu_test <hex_file>             Full CPU-FIFO test (load, run, verify)

  GPU Control:
    gpu_status                      Show GPU status (kernel_done, dma_busy)
    gpu_reset                       Assert GPU reset
    gpu_start                       Release GPU reset (starts executing)
    gpu_stop                        Stop GPU (held in reset)

  GPU Memory:
    gpu_load_imem <hex_file>        Load hex file into GPU IMEM
    gpu_read_imem [start] [count]   Read GPU IMEM (default: 0, 16)
    gpu_read_dmem [start] [count]   Read GPU DMEM (default: 0, 8)
    gpu_write_dmem <addr> <hi> <lo> Write 64-bit GPU DMEM word (hex)

  DMA:
    dma_fifo_to_gpu <f> <g> <len>   FIFO BRAM[f..] -> GPU DMEM[g..] (len words)
    dma_gpu_to_fifo <g> <f> <len>   GPU DMEM[g..] -> FIFO BRAM[f..] (len words)

  Integration Test (requires captured packet in FIFO):
    gpu_test <hex_file>             DMA packet to GPU -> run kernel -> DMA back

  GPU Standalone Tests (PCI-direct, no packet needed):
    test_debug                      Verbose vec_add trace (IMEM + DMEM dumps)
    test_vec_add                    INT16 vector add
    test_vec_sub                    INT16 vector subtract
    test_bf16_mul                   BF16 multiply
    test_bf16_fma                   BF16 fused multiply-add
    test_relu                       INT16 ReLU
    test_ann                        3-layer ReLU ANN (hand-crafted weights)
    test_ann_trained                3-layer ReLU ANN (trained weights)
    test_all                        Run all 7 standalone tests

  Integrated Network Processor Tests (synthetic pkt -> DMA -> GPU -> DMA -> verify):
    test_ann_integrated             ANN via full FIFO->DMA->GPU->DMA->FIFO path
    test_ann_trained_integrated     Trained ANN via full data path
    test_all_integrated             Run all 9 tests (7 standalone + 2 integrated)

  Real Network Packet ANN Tests (requires captured packet, e.g. ARP):
    test_ann_real_pkt               ANN handcrafted on real packet data
    test_ann_trained_real_pkt       ANN trained on real packet data
    test_all_real_pkt               Run both real-packet ANN tests

  Crafted-Input ANN Demo (send known input from node0 via send_bf16.py):
    test_udp_net_a                  f(x) = 3*ReLU(0.5*ReLU(2x-1)+0.25)-0.5
    test_udp_net_b                  f(x) = 2x + 1
    test_udp_both                   Run both networks on same packet""")

def _argint(index, default):
    """Get sys.argv[index] as int, or return default. Python 2.4 safe."""
    if len(sys.argv) > index:
        return int(sys.argv[index])
    return default

def _argstr(index, default):
    """Get sys.argv[index] as string, or return default. Python 2.4 safe."""
    if len(sys.argv) > index:
        return sys.argv[index]
    return default

def _exit_bool(ok):
    """Exit 0 on True, 1 on False. Python 2.4 safe (no ternary)."""
    if ok:
        sys.exit(0)
    else:
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()
        return

    cmd = sys.argv[1]

    # -- FIFO --
    if   cmd == "reset":           cmd_reset()
    elif cmd == "status":          cmd_status()
    elif cmd == "mode":            cmd_mode(sys.argv[2])
    elif cmd == "drain":           cmd_drain()
    elif cmd == "fifo_read":
        s = _argint(2, 0)
        c = _argint(3, 8)
        cmd_fifo_read(s, c)
    elif cmd == "fifo_write":
        cmd_fifo_write(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "bram_write_full":
        cmd_bram_write_full(int(sys.argv[2]), sys.argv[3],
                            sys.argv[4], sys.argv[5])
    elif cmd == "packet_echo":
        cmd_packet_echo(_argstr(2, None))

    # -- CPU --
    elif cmd == "cpu_status":      cmd_cpu_status()
    elif cmd == "cpu_reset":       cmd_cpu_reset()
    elif cmd == "cpu_start":       cmd_cpu_start()
    elif cmd == "cpu_stop":        cmd_cpu_stop()
    elif cmd == "cpu_load_imem":   cmd_cpu_load_imem(sys.argv[2])
    elif cmd == "cpu_read_imem":
        cmd_cpu_read_imem(_argint(2, 0), _argint(3, 16))
    elif cmd == "cpu_read_dmem":
        cmd_cpu_read_dmem(_argint(2, 0), _argint(3, 8))
    elif cmd == "cpu_write_dmem":
        cmd_cpu_write_dmem(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "cpu_la_read":
        cmd_cpu_la_read(_argint(2, 0), _argint(3, 16))
    elif cmd == "cpu_test":        cmd_cpu_test(sys.argv[2])

    # -- GPU --
    elif cmd == "gpu_status":      cmd_gpu_status()
    elif cmd == "gpu_reset":       cmd_gpu_reset()
    elif cmd == "gpu_start":       cmd_gpu_start()
    elif cmd == "gpu_stop":        cmd_gpu_stop()
    elif cmd == "gpu_load_imem":   cmd_gpu_load_imem(sys.argv[2])
    elif cmd == "gpu_read_imem":
        cmd_gpu_read_imem(_argint(2, 0), _argint(3, 16))
    elif cmd == "gpu_read_dmem":
        cmd_gpu_read_dmem(_argint(2, 0), _argint(3, 8))
    elif cmd == "gpu_write_dmem":
        cmd_gpu_write_dmem(int(sys.argv[2]), sys.argv[3], sys.argv[4])

    # -- DMA --
    elif cmd == "dma_fifo_to_gpu":
        cmd_dma_fifo_to_gpu(int(sys.argv[2]), int(sys.argv[3]),
                            int(sys.argv[4]))
    elif cmd == "dma_gpu_to_fifo":
        cmd_dma_gpu_to_fifo(int(sys.argv[2]), int(sys.argv[3]),
                            int(sys.argv[4]))

    # -- Integration test --
    elif cmd == "gpu_test":        cmd_gpu_test(sys.argv[2])

    # -- Standalone GPU tests --
    elif cmd == "test_debug":      test_debug()
    elif cmd == "test_vec_add":    _exit_bool(test_vec_add())
    elif cmd == "test_vec_sub":    _exit_bool(test_vec_sub())
    elif cmd == "test_bf16_mul":   _exit_bool(test_bf16_mul())
    elif cmd == "test_bf16_fma":   _exit_bool(test_bf16_fma())
    elif cmd == "test_relu":       _exit_bool(test_relu())
    elif cmd == "test_ann":        _exit_bool(test_ann())
    elif cmd == "test_ann_trained": _exit_bool(test_ann_trained())
    elif cmd == "test_all":
        t = 0
        t += test_vec_add()
        t += test_vec_sub()
        t += test_bf16_mul()
        t += test_bf16_fma()
        t += test_relu()
        t += test_ann()
        t += test_ann_trained()
        print("=== TOTAL: %d/7 passed ===" % t)
        _exit_bool(t == 7)

    # -- Integrated tests --
    elif cmd == "test_ann_integrated":
        _exit_bool(test_ann_integrated())
    elif cmd == "test_ann_trained_integrated":
        _exit_bool(test_ann_trained_integrated())
    elif cmd == "test_all_integrated":
        _exit_bool(test_all_integrated())

    # -- Real packet ANN tests --
    elif cmd == "test_ann_real_pkt":
        _exit_bool(test_ann_real_pkt())
    elif cmd == "test_ann_trained_real_pkt":
        _exit_bool(test_ann_trained_real_pkt())
    elif cmd == "test_all_real_pkt":
        _exit_bool(test_all_real_pkt())

    # -- Crafted-input UDP ANN demo --
    elif cmd == "test_udp_net_a":
        _exit_bool(test_udp_net_a())
    elif cmd == "test_udp_net_b":
        _exit_bool(test_udp_net_b())
    elif cmd == "test_udp_both":
        _exit_bool(test_udp_both())

    else:
        print("Unknown command: %s" % cmd)
        usage()

if __name__ == "__main__":
    main()
