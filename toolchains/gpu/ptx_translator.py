#!/usr/bin/env python3
"""
ptx_translator.py — Translate parsed PTX kernels to custom GPU assembly.

Translation strategy:
  nvcc generates 4 scalar operations per kernel (one per int16/bf16 element).
  Our GPU has 4-lane SIMD — one instruction processes all 4 elements.
  So the translator must:
    1. Identify the core compute operation (add.s16, sub.s16, max.s16, fma.rn.bf16)
    2. Count input/output array parameters
    3. Detect FMA-as-multiply pattern (fma with c=-0.0)
    4. Collapse 4 scalar iterations into 1 SIMD instruction
    5. Emit our custom assembly

PTX pattern (all 5 kernels):
  [ld.param] [tid.x] [bounds check] [addr math] [4x: ld, compute, st] [ret]
  ->
  Our assembly: [LD inputs] [compute] [ST output] [HALT]

Memory layout convention:
  Kernel params map to consecutive DMEM word addresses starting at 0.
  param_0 -> DMEM[0], param_1 -> DMEM[1], etc.
  (The last u32 param is n_elements, not a memory pointer.)

Command example: python3 .\ptx_translator.py ..\kernels\kernel.ptx ..\programs\translated\
"""

import sys
from ptx_parser import KernelInfo, parse_ptx


# ---------------------------------------------------------------------------
# Compute operation detection
# ---------------------------------------------------------------------------

# Map PTX compute ops -> our ISA mnemonic + dtype
PTX_TO_ISA = {
    'add.s16':     ('ADD',  'S16'),
    'sub.s16':     ('SUB',  'S16'),
    'max.s16':     ('RELU', 'S16'),    # max(x, 0) = ReLU
    'fma.rn.bf16': ('FMA',  'BF16'),   # may be MUL if c=-0.0
}


def _find_compute_ops(kernel):
    """Find the core compute operations in a kernel (exclude boilerplate)."""
    compute_ops = []
    for instr in kernel.instructions:
        if instr.op in PTX_TO_ISA:
            compute_ops.append(instr)
    return compute_ops


def _is_fma_as_mul(kernel):
    """Detect if FMA is used as multiply (c = -0.0 = 0x8000).

    nvcc implements __hmul(a,b) as fma.rn.bf16(a, b, -0.0).
    We detect this by finding mov.b16 c, 0x8000U before fma.rn.bf16.
    """
    for instr in kernel.instructions:
        if instr.op == 'mov.b16' and len(instr.operands) >= 2:
            if '0X8000' in instr.operands[1].upper():
                return True
    return False


def _count_array_params(kernel):
    """Count u64 params (array pointers) vs u32 params (scalars like n_elements)."""
    arrays = [p for p in kernel.params if p.type == 'u64']
    scalars = [p for p in kernel.params if p.type == 'u32']
    return len(arrays), len(scalars)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_kernel(kernel):
    """Translate a single parsed PTX kernel to our custom assembly text.

    Returns:
        tuple: (kernel_name, asm_text, translation_notes)
    """
    name = kernel.name
    n_arrays, n_scalars = _count_array_params(kernel)
    compute_ops = _find_compute_ops(kernel)
    notes = []

    if not compute_ops:
        # Special case: relu_bf16 uses branching in PTX but our HW has native RELU
        if 'relu' in name.lower() and 'bf16' in name.lower():
            notes.append("PTX uses branching for BF16 ReLU; our HW has native RELU BF16")
            primary_op = 'relu_bf16_special'
            isa_op, dtype = 'RELU', 'BF16'
        else:
            notes.append("WARNING: No recognized compute operations found")
            return name, f"# {name} -- no translatable ops\n    HALT\n", notes
    else:
        # Identify the core operation
        primary_op = compute_ops[0].op
        isa_op, dtype = PTX_TO_ISA[primary_op]

    # Check for FMA-as-multiply pattern
    if primary_op == 'fma.rn.bf16' and _is_fma_as_mul(kernel):
        isa_op = 'MUL'
        notes.append("Detected __hmul pattern: fma.rn.bf16(a, b, -0.0) -> MUL BF16")

    # Determine how many input arrays vs output arrays
    # Convention: last u64 param is output (or last before the u32 n_elements)
    # For FMA: 3 inputs + 1 output = 4 arrays
    if isa_op == 'FMA':
        n_inputs = n_arrays - 1  # 3 inputs, 1 output
    elif isa_op == 'RELU':
        n_inputs = 1  # 1 input, 1 output
    else:
        n_inputs = n_arrays - 1  # typically 2 inputs, 1 output

    output_addr = n_inputs  # output DMEM address = number of inputs

    notes.append(f"PTX op: {primary_op} -> ISA op: {isa_op} {dtype}")
    notes.append(f"Arrays: {n_inputs} inputs + 1 output, DMEM[0..{output_addr}]")
    notes.append(f"4 scalar PTX ops collapsed to 1 SIMD instruction")

    # Generate assembly
    lines = [f"# {name}.asm — Auto-translated from PTX"]
    lines.append(f"# Source: kernel.cu -> nvcc -ptx -> ptx_translator.py")
    lines.append(f"# PTX compute op: {primary_op} -> {isa_op} {dtype}")
    lines.append(f"# {n_inputs} input array(s) at DMEM[0..{n_inputs-1}], output at DMEM[{output_addr}]")
    lines.append(f"# 4 scalar PTX ops collapsed to 1 SIMD instruction (4 x 16-bit lanes)")
    lines.append("")

    # Load inputs
    reg_num = 1
    input_regs = []
    for i in range(n_inputs):
        lines.append(f"    LD   R{reg_num}, {i}(R0)")
        input_regs.append(f"R{reg_num}")
        reg_num += 1

    # Compute
    result_reg = f"R{reg_num}"
    if isa_op == 'RELU':
        dtype_suffix = f", {dtype}" if dtype != 'S16' else ""
        lines.append(f"    RELU {result_reg}, {input_regs[0]}{dtype_suffix}")
    elif isa_op == 'FMA':
        lines.append(f"    FMA  {result_reg}, {input_regs[0]}, {input_regs[1]}, {input_regs[2]}")
    elif isa_op == 'MUL':
        lines.append(f"    MUL  {result_reg}, {input_regs[0]}, {input_regs[1]}, {dtype}")
    elif isa_op in ('ADD', 'SUB'):
        lines.append(f"    {isa_op}  {result_reg}, {input_regs[0]}, {input_regs[1]}")
    else:
        lines.append(f"    {isa_op}  {result_reg}, {', '.join(input_regs)}")

    # Store output
    lines.append(f"    ST   {result_reg}, {output_addr}(R0)")

    # Halt
    lines.append("    HALT")
    lines.append("")

    return name, '\n'.join(lines), notes


def translate_all(kernels):
    """Translate all kernels from a PTX file.

    Returns:
        list of (name, asm_text, notes) tuples
    """
    results = []
    for kernel in kernels:
        results.append(translate_kernel(kernel))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ptx_translator.py input.ptx [output_dir]")
        print("  Translates all kernels in input.ptx to .asm files")
        sys.exit(1)

    import os
    from ptx_parser import parse_ptx

    ptx_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    with open(ptx_path, 'r') as f:
        text = f.read()

    kernels = parse_ptx(text)
    results = translate_all(kernels)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Translated {len(results)} kernels from {ptx_path}")
    print("=" * 60)

    for name, asm_text, notes in results:
        out_path = os.path.join(output_dir, f"{name}.asm")
        with open(out_path, 'w') as f:
            f.write(asm_text)

        print(f"\n  {name} -> {out_path}")
        for note in notes:
            print(f"    {note}")
        print(f"  --- Assembly ---")
        for line in asm_text.strip().split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                print(f"    {line}")
