#!/usr/bin/env python3
"""
compile.py — Full compilation pipeline for custom FPGA GPU.

Pipeline:
    kernel.cu -> nvcc -ptx -> kernel.ptx -> ptx_parser -> ptx_translator -> .asm -> assembler -> .hex

Usage:
    python compile.py <input.cu|input.ptx> [-o output_dir] [--kernel name] [--skip-nvcc]

If input is .cu, runs nvcc first (requires CUDA toolkit).
If input is .ptx, skips nvcc step.
If --kernel is specified, only compile that kernel.
"""

import argparse
import os
import subprocess
import sys

from ptx_parser import parse_ptx
from ptx_translator import translate_all
from assembler import assemble_string


def run_nvcc(cu_path, ptx_path):
    """Run nvcc to compile .cu -> .ptx"""
    cmd = ['nvcc', '--machine', '64', '-ptx', '-arch=sm_80', cu_path, '-o', ptx_path]

    # Try to find cl.exe for Windows (MSVC required by nvcc)
    msvc_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    ]
    for base in msvc_paths:
        if os.path.isdir(base):
            versions = sorted(os.listdir(base), reverse=True)
            if versions:
                ccbin = os.path.join(base, versions[0], 'bin', 'Hostx64', 'x64')
                if os.path.isdir(ccbin):
                    cmd.insert(3, f'-ccbin')
                    cmd.insert(4, ccbin)
                    break

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  nvcc FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()}")
            return False
        print(f"  nvcc OK -> {ptx_path}")
        return True
    except FileNotFoundError:
        print("  ERROR: nvcc not found. Install CUDA toolkit or use --skip-nvcc with a .ptx file.")
        return False
    except subprocess.TimeoutExpired:
        print("  ERROR: nvcc timed out")
        return False


def compile_pipeline(input_path, output_dir, kernel_filter=None, skip_nvcc=False):
    """Run the full compilation pipeline."""
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()

    # Step 1: .cu -> .ptx (if needed)
    if ext == '.cu' and not skip_nvcc:
        ptx_path = os.path.splitext(input_path)[0] + '.ptx'
        print(f"\n[Step 1] Compiling CUDA -> PTX")
        if not run_nvcc(input_path, ptx_path):
            # Fall back to existing .ptx if available
            if os.path.exists(ptx_path):
                print(f"  Using existing {ptx_path}")
            else:
                print("  FATAL: No .ptx file available")
                return False
    elif ext == '.ptx' or skip_nvcc:
        if ext == '.cu':
            ptx_path = os.path.splitext(input_path)[0] + '.ptx'
        else:
            ptx_path = input_path
        print(f"\n[Step 1] Skipping nvcc, using existing PTX: {ptx_path}")
    else:
        print(f"  ERROR: Unsupported input format: {ext}")
        return False

    if not os.path.exists(ptx_path):
        print(f"  FATAL: PTX file not found: {ptx_path}")
        return False

    # Step 2: Parse PTX
    print(f"\n[Step 2] Parsing PTX")
    with open(ptx_path, 'r') as f:
        ptx_text = f.read()
    kernels = parse_ptx(ptx_text)
    print(f"  Found {len(kernels)} kernels: {[k.name for k in kernels]}")

    # Step 3: Translate PTX -> assembly
    print(f"\n[Step 3] Translating PTX -> custom assembly")
    results = translate_all(kernels)

    # Step 4: Assemble -> hex
    print(f"\n[Step 4] Assembling -> hex")
    compiled = []
    for name, asm_text, notes in results:
        if kernel_filter and name != kernel_filter:
            continue

        print(f"\n  --- {name} ---")
        for note in notes:
            print(f"    {note}")

        # Write .asm file
        asm_path = os.path.join(output_dir, f"{name}.asm")
        with open(asm_path, 'w') as f:
            f.write(asm_text)

        # Assemble to hex
        try:
            encoded = assemble_string(asm_text)
            hex_path = os.path.join(output_dir, f"{name}.hex")
            with open(hex_path, 'w') as f:
                for word in encoded:
                    f.write(f"{word:08X}\n")
            print(f"    -> {hex_path} ({len(encoded)} instructions)")
            for i, word in enumerate(encoded):
                print(f"       [{i}] {word:08X}")
            compiled.append((name, hex_path, len(encoded)))
        except Exception as e:
            print(f"    ASSEMBLY ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Compilation Summary")
    print(f"{'='*60}")
    print(f"  Input:   {input_path}")
    print(f"  PTX:     {ptx_path}")
    print(f"  Output:  {output_dir}/")
    print(f"  Kernels compiled: {len(compiled)}")
    for name, hex_path, n_instr in compiled:
        print(f"    {name}: {n_instr} instructions -> {os.path.basename(hex_path)}")
    print(f"{'='*60}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compile CUDA/PTX to custom GPU hex')
    parser.add_argument('input', help='Input .cu or .ptx file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--kernel', default=None,
                        help='Only compile specific kernel by name')
    parser.add_argument('--skip-nvcc', action='store_true',
                        help='Skip nvcc step (use existing .ptx)')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.input))

    success = compile_pipeline(args.input, args.output,
                               kernel_filter=args.kernel,
                               skip_nvcc=args.skip_nvcc)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
