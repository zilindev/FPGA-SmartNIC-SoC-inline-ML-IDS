# FPGA SmartNIC SoC for Inline ML IDS

An FPGA-based SmartNIC prototype that performs machine-learning intrusion detection directly in the network data path. The design targets the NetFPGA V2 platform and combines an ARM-compatible processor, a custom SIMD GPU, packet buffering, and DMA in a single FPGA system.

## Highlights

- Four-thread ARM-compatible processor for control and packet orchestration
- Custom four-lane SIMD GPU with BF16 and INT16 arithmetic
- Six-stage GPU pipeline with an extended broadcast instruction for packed inference
- Convertible FIFO implemented with dual-port BRAM for network and processor access
- DMA engine for transferring packet data between FIFO memory and GPU data memory
- Packetized inference using an 11-16-8-2 multilayer perceptron IDS model
- Hardware cycle counter and host-side tools for latency and throughput measurement
- Validated through RTL simulation and deployment on a NetFPGA V2 (Virtex-II Pro XC2VP50)

## Data Flow

```text
Ethernet packet -> Convertible FIFO -> ARM orchestration -> DMA
                -> SIMD GPU inference -> IDS result -> response packet
```

## Repository Layout

- `src/` - Synthesizable Verilog-2001 RTL for the ARM processor, GPU, FIFO, DMA, and top-level integration
- `include/` - Hardware parameters, register definitions, and NetFPGA integration files
- `programs/` - ARM firmware, GPU kernels, trained weights, and expected inference data
- `scripts/` - Packet generation, model training, hardware control, and performance analysis tools
- `toolchains/` - ARM utilities and the custom GPU assembler/compiler toolchain
- `tb/` - RTL testbenches and simulation primitives
- `synth/` - NetFPGA synthesis inputs for Xilinx ISE

## Technology

Verilog-2001, Python, custom SIMD ISA, BF16/INT16 arithmetic, Xilinx ISE 10.1, Icarus Verilog, and NetFPGA V2.
