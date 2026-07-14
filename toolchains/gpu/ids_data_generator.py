#!/usr/bin/env python3
"""
ids_data_generator.py -- Generate DMEM hex for IDS MLP (11->16->8->2).

Produces $readmemh-compatible .hex file with all values replicated across
4 BF16 SIMD lanes (scalar mode: {v, v, v, v}).

DMEM layout (383 words):
  [0..10]     Input features x[0..10]
  [11..186]   L1 weights w1[n][i], neuron-major (16 neurons x 11 inputs)
  [187..202]  L1 biases b1[0..15]
  [203..218]  L1 output (zeroed, filled by GPU)
  [219..346]  L2 weights w2[n][i] (8 neurons x 16 inputs)
  [347..354]  L2 biases b2[0..7]
  [355..362]  L2 output (zeroed)
  [363..378]  L3 weights w3[n][i] (2 neurons x 8 inputs)
  [379..380]  L3 biases b3[0..1]
  [381..382]  L3 output (zeroed)

Usage:
    python ids_data_generator.py [-o output.hex] [--test] [--weights weights.json]
"""

import sys
import os
import json
import struct

# Add parent path for bf16_utils
sys.path.insert(0, os.path.dirname(__file__))
from bf16_utils import float_to_bf16, bf16_to_float, pack_bf16_vector, format_hex64


# Network dimensions
L1_IN, L1_OUT = 11, 16
L2_IN, L2_OUT = 16, 8
L3_IN, L3_OUT = 8, 2

# DMEM addresses
X_BASE   = 0
W1_BASE  = 11
B1_BASE  = 187
H1_BASE  = 203
W2_BASE  = 219
B2_BASE  = 347
H2_BASE  = 355
W3_BASE  = 363
B3_BASE  = 379
OUT_BASE = 381
DMEM_SIZE = 383


def replicate_bf16(value):
    """Pack a single float as replicated BF16 across all 4 SIMD lanes."""
    return pack_bf16_vector([value, value, value, value])


def bf16_fma(a, b, c):
    """Simulate BF16 FMA: truncate(float(a) * float(b) + float(c)) -> BF16.
    Matches hardware: multiply in extended precision, add, then truncate."""
    fa = bf16_to_float(float_to_bf16(a))
    fb = bf16_to_float(float_to_bf16(b))
    fc = bf16_to_float(float_to_bf16(c))
    result = fa * fb + fc
    # Truncate to BF16 and back
    return bf16_to_float(float_to_bf16(result))


def bf16_relu(x):
    """BF16 ReLU: max(0, x)."""
    return max(0.0, x)


def generate_test_weights():
    """Generate simple test weights for verification.
    Uses small known values so results are easy to verify by hand."""
    import random
    random.seed(42)

    weights = {
        'w1': [[random.uniform(-0.5, 0.5) for _ in range(L1_IN)] for _ in range(L1_OUT)],
        'b1': [random.uniform(-0.1, 0.1) for _ in range(L1_OUT)],
        'w2': [[random.uniform(-0.5, 0.5) for _ in range(L2_IN)] for _ in range(L2_OUT)],
        'b2': [random.uniform(-0.1, 0.1) for _ in range(L2_OUT)],
        'w3': [[random.uniform(-0.5, 0.5) for _ in range(L3_IN)] for _ in range(L3_OUT)],
        'b3': [random.uniform(-0.1, 0.1) for _ in range(L3_OUT)],
    }
    return weights


def generate_simple_test_weights():
    """All weights = 0.25, all biases = 0. Easy to verify."""
    return {
        'w1': [[0.25] * L1_IN for _ in range(L1_OUT)],
        'b1': [0.0] * L1_OUT,
        'w2': [[0.25] * L2_IN for _ in range(L2_OUT)],
        'b2': [0.0] * L2_OUT,
        'w3': [[0.25] * L3_IN for _ in range(L3_OUT)],
        'b3': [0.0] * L3_OUT,
    }


def compute_reference(inputs, weights):
    """Compute expected output using BF16-quantized arithmetic.
    Simulates the GPU's FMA-based computation step by step."""

    # Quantize inputs
    x = [bf16_to_float(float_to_bf16(v)) for v in inputs]

    # Layer 1: h1 = ReLU(W1 @ x + b1)
    h1 = []
    for n in range(L1_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b1'][n]))
        for i in range(L1_IN):
            w = bf16_to_float(float_to_bf16(weights['w1'][n][i]))
            acc = bf16_fma(w, x[i], acc)
        h1.append(bf16_relu(acc))

    # Layer 2: h2 = ReLU(W2 @ h1 + b2)
    h2 = []
    for n in range(L2_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b2'][n]))
        for i in range(L2_IN):
            w = bf16_to_float(float_to_bf16(weights['w2'][n][i]))
            acc = bf16_fma(w, h1[i], acc)
        h2.append(bf16_relu(acc))

    # Layer 3: out = W3 @ h2 + b3 (no activation)
    out = []
    for n in range(L3_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b3'][n]))
        for i in range(L3_IN):
            w = bf16_to_float(float_to_bf16(weights['w3'][n][i]))
            acc = bf16_fma(w, h2[i], acc)
        out.append(acc)

    return h1, h2, out


def build_dmem(inputs, weights):
    """Build the full 383-word DMEM array as 64-bit values."""
    dmem = [0] * DMEM_SIZE

    # Inputs: DMEM[0..10]
    for i in range(L1_IN):
        dmem[X_BASE + i] = replicate_bf16(inputs[i])

    # L1 weights: DMEM[11..186], neuron-major
    for n in range(L1_OUT):
        for i in range(L1_IN):
            dmem[W1_BASE + n * L1_IN + i] = replicate_bf16(weights['w1'][n][i])

    # L1 biases: DMEM[187..202]
    for n in range(L1_OUT):
        dmem[B1_BASE + n] = replicate_bf16(weights['b1'][n])

    # L1 outputs: zeroed (DMEM[203..218])
    # L2 weights: DMEM[219..346]
    for n in range(L2_OUT):
        for i in range(L2_IN):
            dmem[W2_BASE + n * L2_IN + i] = replicate_bf16(weights['w2'][n][i])

    # L2 biases: DMEM[347..354]
    for n in range(L2_OUT):
        dmem[B2_BASE + n] = replicate_bf16(weights['b2'][n])

    # L2 outputs: zeroed (DMEM[355..362])
    # L3 weights: DMEM[363..378]
    for n in range(L3_OUT):
        for i in range(L3_IN):
            dmem[W3_BASE + n * L3_IN + i] = replicate_bf16(weights['w3'][n][i])

    # L3 biases: DMEM[379..380]
    for n in range(L3_OUT):
        dmem[B3_BASE + n] = replicate_bf16(weights['b3'][n])

    # L3 outputs: zeroed (DMEM[381..382])
    return dmem


def write_hex(dmem, path):
    """Write DMEM array to $readmemh-compatible hex file."""
    with open(path, 'w') as f:
        for word in dmem:
            f.write(format_hex64(word) + '\n')


def main():
    output_path = 'data_ids_11_16_8_2.hex'
    use_simple = False
    weights_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '-o' and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == '--test':
            use_simple = True
            i += 1
        elif args[i] == '--weights' and i + 1 < len(args):
            weights_path = args[i + 1]
            i += 2
        else:
            i += 1

    # Load or generate weights
    if weights_path:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        print(f"Loaded weights from {weights_path}")
    elif use_simple:
        weights = generate_simple_test_weights()
        print("Using simple test weights (all 0.25, biases 0)")
    else:
        weights = generate_test_weights()
        print("Using random test weights (seed=42)")

    # Test inputs
    inputs = [1.0] * L1_IN
    print(f"Test inputs: {inputs}")

    # Compute reference output
    h1, h2, out = compute_reference(inputs, weights)
    print(f"\nReference computation (BF16-quantized):")
    print(f"  Layer 1 output (h1): {['%.4f' % v for v in h1]}")
    print(f"  Layer 2 output (h2): {['%.4f' % v for v in h2]}")
    print(f"  Layer 3 output (out): {['%.4f' % v for v in out]}")

    # Expected output as hex
    print(f"\nExpected output DMEM[{OUT_BASE}..{OUT_BASE + L3_OUT - 1}]:")
    for n in range(L3_OUT):
        bf = float_to_bf16(out[n])
        word = replicate_bf16(out[n])
        print(f"  out[{n}] = {out[n]:.4f} -> BF16 0x{bf:04X} -> DMEM 0x{format_hex64(word)}")

    # Build and write DMEM
    dmem = build_dmem(inputs, weights)
    write_hex(dmem, output_path)
    print(f"\nWrote {len(dmem)} DMEM words to {output_path}")

    # Also save weights as JSON for reproducibility
    json_path = output_path.replace('.hex', '_weights.json')
    with open(json_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Saved weights to {json_path}")

    # Save expected outputs for testbench verification
    expected_path = output_path.replace('.hex', '_expected.txt')
    with open(expected_path, 'w') as f:
        f.write("# Expected DMEM values after kernel execution\n")
        f.write("# Format: DMEM_ADDR HEX_VALUE FLOAT_VALUE\n")
        for n in range(L1_OUT):
            word = replicate_bf16(h1[n])
            f.write(f"{H1_BASE + n} {format_hex64(word)} {h1[n]:.6f}\n")
        for n in range(L2_OUT):
            word = replicate_bf16(h2[n])
            f.write(f"{H2_BASE + n} {format_hex64(word)} {h2[n]:.6f}\n")
        for n in range(L3_OUT):
            word = replicate_bf16(out[n])
            f.write(f"{OUT_BASE + n} {format_hex64(word)} {out[n]:.6f}\n")
    print(f"Saved expected values to {expected_path}")


if __name__ == '__main__':
    main()
