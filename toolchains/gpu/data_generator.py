#!/usr/bin/env python3
"""
data_generator.py — Generate data hex files and expected outputs for all 5 kernels.
Produces $readmemh-compatible 64-bit data files for DMEM initialization.

Usage:
    python data_generator.py [output_dir]
    Default output_dir: ../programs/
"""

import sys
import os
from bf16_utils import (
    float_to_bf16, bf16_to_float, pack_bf16_vector, unpack_bf16_vector,
    pack_int16_vector, unpack_int16_vector, format_hex64,
)


def write_data_hex(filepath, words, comments=None):
    """Write a list of 64-bit words to a hex file (one per line)."""
    with open(filepath, 'w') as f:
        for i, word in enumerate(words):
            line = format_hex64(word)
            if comments and i < len(comments):
                line += f"  // {comments[i]}"
            f.write(line + "\n")


def generate_vec_add_i16(outdir):
    """vec_add_i16: C[i] = A[i] + B[i], int16, 4 lanes."""
    a_vals = [1, 2, 3, 4]
    b_vals = [5, 6, 7, 8]
    c_vals = [a + b for a, b in zip(a_vals, b_vals)]  # [6, 8, 10, 12]

    a_packed = pack_int16_vector(a_vals)
    b_packed = pack_int16_vector(b_vals)
    c_packed = pack_int16_vector(c_vals)

    # DMEM layout: [0]=A, [1]=B, [2]=result (cleared)
    data = [a_packed, b_packed, 0]
    comments = [
        f"A = {a_vals}",
        f"B = {b_vals}",
        "result (cleared)",
    ]
    write_data_hex(os.path.join(outdir, "data_vec_add_i16.hex"), data, comments)

    print(f"  vec_add_i16: A={a_vals} + B={b_vals} = {c_vals}")
    print(f"    Expected DMEM[2] = {format_hex64(c_packed)}")
    return c_packed


def generate_vec_sub_i16(outdir):
    """vec_sub_i16: C[i] = A[i] - B[i], int16, 4 lanes."""
    a_vals = [10, 20, 30, 40]
    b_vals = [3, 5, 10, 15]
    c_vals = [a - b for a, b in zip(a_vals, b_vals)]  # [7, 15, 20, 25]

    a_packed = pack_int16_vector(a_vals)
    b_packed = pack_int16_vector(b_vals)
    c_packed = pack_int16_vector(c_vals)

    data = [a_packed, b_packed, 0]
    comments = [
        f"A = {a_vals}",
        f"B = {b_vals}",
        "result (cleared)",
    ]
    write_data_hex(os.path.join(outdir, "data_vec_sub_i16.hex"), data, comments)

    print(f"  vec_sub_i16: A={a_vals} - B={b_vals} = {c_vals}")
    print(f"    Expected DMEM[2] = {format_hex64(c_packed)}")
    return c_packed


def generate_relu_i16(outdir):
    """relu_i16: out[i] = max(0, in[i]), int16, 4 lanes."""
    in_vals = [-3, 5, -1, 7]
    out_vals = [max(0, x) for x in in_vals]  # [0, 5, 0, 7]

    in_packed = pack_int16_vector(in_vals)
    out_packed = pack_int16_vector(out_vals)

    # DMEM layout: [0]=input, [1]=result (cleared)
    data = [in_packed, 0]
    comments = [
        f"in = {in_vals}",
        "result (cleared)",
    ]
    write_data_hex(os.path.join(outdir, "data_relu_i16.hex"), data, comments)

    print(f"  relu_i16: max(0, {in_vals}) = {out_vals}")
    print(f"    Expected DMEM[1] = {format_hex64(out_packed)}")
    return out_packed


def generate_bf16_mul(outdir):
    """bf16_mul: C[i] = A[i] * B[i], BF16, 4 lanes."""
    a_vals = [1.0, 2.0, 0.5, 3.0]
    b_vals = [2.0, 3.0, 0.5, 0.25]
    # Expected: [2.0, 6.0, 0.25, 0.75]
    c_vals = [a * b for a, b in zip(a_vals, b_vals)]

    # Verify through BF16 path (convert to bf16 and back to check precision)
    c_bf16 = []
    for a, b in zip(a_vals, b_vals):
        # Simulate: truncate each operand to BF16, multiply, truncate result
        result = bf16_to_float(float_to_bf16(a)) * bf16_to_float(float_to_bf16(b))
        c_bf16.append(bf16_to_float(float_to_bf16(result)))

    a_packed = pack_bf16_vector(a_vals)
    b_packed = pack_bf16_vector(b_vals)
    c_packed = pack_bf16_vector(c_bf16)

    data = [a_packed, b_packed, 0]
    comments = [
        f"A = {a_vals} (BF16)",
        f"B = {b_vals} (BF16)",
        "result (cleared)",
    ]
    write_data_hex(os.path.join(outdir, "data_bf16_mul.hex"), data, comments)

    print(f"  bf16_mul: A={a_vals} * B={b_vals}")
    print(f"    Expected (BF16): {c_bf16}")
    print(f"    Expected DMEM[2] = {format_hex64(c_packed)}")
    return c_packed


def generate_bf16_fma(outdir):
    """bf16_fma: D[i] = A[i]*B[i] + C[i], BF16, 4 lanes."""
    a_vals = [1.0, 1.5, 2.0, 0.5]
    b_vals = [2.0, 1.5, 3.0, 4.0]
    c_vals = [3.0, 0.25, 1.0, 0.5]
    # Expected: [5.0, 2.5, 7.0, 2.5]
    d_vals = [a * b + c for a, b, c in zip(a_vals, b_vals, c_vals)]

    # BF16-accurate computation
    d_bf16 = []
    for a, b, c in zip(a_vals, b_vals, c_vals):
        af = bf16_to_float(float_to_bf16(a))
        bf_ = bf16_to_float(float_to_bf16(b))
        cf = bf16_to_float(float_to_bf16(c))
        result = af * bf_ + cf
        d_bf16.append(bf16_to_float(float_to_bf16(result)))

    a_packed = pack_bf16_vector(a_vals)
    b_packed = pack_bf16_vector(b_vals)
    c_packed = pack_bf16_vector(c_vals)
    d_packed = pack_bf16_vector(d_bf16)

    # DMEM layout: [0]=A, [1]=B, [2]=C, [3]=result (cleared)
    data = [a_packed, b_packed, c_packed, 0]
    comments = [
        f"A = {a_vals} (BF16)",
        f"B = {b_vals} (BF16)",
        f"C = {c_vals} (BF16)",
        "result (cleared)",
    ]
    write_data_hex(os.path.join(outdir, "data_bf16_fma.hex"), data, comments)

    print(f"  bf16_fma: A={a_vals} * B={b_vals} + C={c_vals}")
    print(f"    Expected (BF16): {d_bf16}")
    print(f"    Expected DMEM[3] = {format_hex64(d_packed)}")
    return d_packed


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'programs')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating data hex files in: {outdir}")
    print("=" * 60)

    generate_vec_add_i16(outdir)
    print()
    generate_vec_sub_i16(outdir)
    print()
    generate_relu_i16(outdir)
    print()
    generate_bf16_mul(outdir)
    print()
    generate_bf16_fma(outdir)

    print()
    print("=" * 60)
    print("All 5 kernel data files generated.")


if __name__ == '__main__':
    main()
