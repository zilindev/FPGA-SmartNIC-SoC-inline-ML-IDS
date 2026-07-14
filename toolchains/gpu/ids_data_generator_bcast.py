#!/usr/bin/env python3
"""
ids_data_generator_bcast.py -- Generate packed DMEM hex for BCAST-optimized
IDS MLP kernel (11->16->8->2).

Differs from ids_data_generator.py in that weights and biases are packed
4-per-word (or 2-per-word for L3) so the kernel can use all 4 SIMD lanes in
parallel instead of scalar mode. Inputs and layer outputs are kept scalar
(replicated across all 4 lanes) so FMA with packed weights and replicated
inputs naturally produces 4 independent neuron accumulators.

Packed DMEM layout (128 words):
  [0..10]     Input x[0..10] (scalar, replicated)            11 words
  [11..54]    L1 weights packed: 4 groups x 11 inputs        44 words
              word [11 + g*11 + i] = {w1[4g+3][i], w1[4g+2][i],
                                       w1[4g+1][i], w1[4g+0][i]}
  [55..58]    L1 biases packed: 4 words                       4 words
              [55] = {b1[3], b1[2], b1[1], b1[0]}
              [56] = {b1[7], b1[6], b1[5], b1[4]}
              [57] = {b1[11],b1[10],b1[9], b1[8]}
              [58] = {b1[15],b1[14],b1[13],b1[12]}
  [59..74]    L1 outputs h1[0..15] (scalar, written by kernel)
  [75..106]   L2 weights packed: 2 groups x 16 inputs        32 words
              word [75 + g*16 + i] = {w2[4g+3][i], w2[4g+2][i],
                                       w2[4g+1][i], w2[4g+0][i]}
  [107..108]  L2 biases packed: 2 words                       2 words
  [109..116]  L2 outputs h2[0..7] (scalar, written by kernel)
  [117..124]  L3 weights packed 2-per-word: 1 group x 8 inps  8 words
              word [117+i] = {0, 0, w3[1][i], w3[0][i]}
  [125]       L3 biases packed: {0, 0, b3[1], b3[0]}          1 word
  [126..127]  L3 outputs out[0..1] (scalar, written by kernel)

Total: 128 words (3.0x reduction from 383-word scalar layout).

Usage:
    python ids_data_generator_bcast.py [-o output.hex] [--test] [--weights w.json]
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from bf16_utils import float_to_bf16, bf16_to_float, pack_bf16_vector, format_hex64


# Network dimensions
L1_IN, L1_OUT = 11, 16
L2_IN, L2_OUT = 16, 8
L3_IN, L3_OUT = 8, 2

# DMEM addresses
X_BASE    = 0
W1_BASE   = 11
B1_BASE   = 55
H1_BASE   = 59
W2_BASE   = 75
B2_BASE   = 107
H2_BASE   = 109
W3_BASE   = 117
B3_BASE   = 125
OUT_BASE  = 126
DMEM_SIZE = 128


def replicate_bf16(value):
    """Pack a single float as replicated BF16 across all 4 SIMD lanes."""
    return pack_bf16_vector([value, value, value, value])


def pack4_bf16(v3, v2, v1, v0):
    """Pack 4 floats into one 64-bit word.

    Lane layout matches pack_bf16_vector: args are [lane3, lane2, lane1, lane0]
    where lane 0 is the LSB. This is the layout that `BCAST Rd, Rs, k` reads
    (k=0 selects lane 0 = LSB).
    """
    return pack_bf16_vector([v3, v2, v1, v0])


def bf16_fma(a, b, c):
    """Simulate BF16 FMA: truncate(float(a) * float(b) + float(c)) -> BF16."""
    fa = bf16_to_float(float_to_bf16(a))
    fb = bf16_to_float(float_to_bf16(b))
    fc = bf16_to_float(float_to_bf16(c))
    return bf16_to_float(float_to_bf16(fa * fb + fc))


def bf16_relu(x):
    return max(0.0, x)


def generate_simple_test_weights():
    """All weights = 0.25, all biases = 0. Easy hand-verification."""
    return {
        'w1': [[0.25] * L1_IN for _ in range(L1_OUT)],
        'b1': [0.0] * L1_OUT,
        'w2': [[0.25] * L2_IN for _ in range(L2_OUT)],
        'b2': [0.0] * L2_OUT,
        'w3': [[0.25] * L3_IN for _ in range(L3_OUT)],
        'b3': [0.0] * L3_OUT,
    }


def generate_test_weights():
    """Random weights for broader verification (same seed as scalar generator)."""
    import random
    random.seed(42)
    return {
        'w1': [[random.uniform(-0.5, 0.5) for _ in range(L1_IN)] for _ in range(L1_OUT)],
        'b1': [random.uniform(-0.1, 0.1) for _ in range(L1_OUT)],
        'w2': [[random.uniform(-0.5, 0.5) for _ in range(L2_IN)] for _ in range(L2_OUT)],
        'b2': [random.uniform(-0.1, 0.1) for _ in range(L2_OUT)],
        'w3': [[random.uniform(-0.5, 0.5) for _ in range(L3_IN)] for _ in range(L3_OUT)],
        'b3': [random.uniform(-0.1, 0.1) for _ in range(L3_OUT)],
    }


def compute_reference(inputs, weights):
    """Compute expected outputs using BF16-quantized arithmetic."""
    x = [bf16_to_float(float_to_bf16(v)) for v in inputs]

    h1 = []
    for n in range(L1_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b1'][n]))
        for i in range(L1_IN):
            w = bf16_to_float(float_to_bf16(weights['w1'][n][i]))
            acc = bf16_fma(w, x[i], acc)
        h1.append(bf16_relu(acc))

    h2 = []
    for n in range(L2_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b2'][n]))
        for i in range(L2_IN):
            w = bf16_to_float(float_to_bf16(weights['w2'][n][i]))
            acc = bf16_fma(w, h1[i], acc)
        h2.append(bf16_relu(acc))

    out = []
    for n in range(L3_OUT):
        acc = bf16_to_float(float_to_bf16(weights['b3'][n]))
        for i in range(L3_IN):
            w = bf16_to_float(float_to_bf16(weights['w3'][n][i]))
            acc = bf16_fma(w, h2[i], acc)
        out.append(acc)

    return h1, h2, out


def build_dmem(inputs, weights):
    """Build the 128-word packed DMEM array."""
    dmem = [0] * DMEM_SIZE

    # Inputs: scalar replicated
    for i in range(L1_IN):
        dmem[X_BASE + i] = replicate_bf16(inputs[i])

    # L1 weights: 4 groups of 4 neurons, each group has L1_IN input-words
    for g in range(L1_OUT // 4):
        for i in range(L1_IN):
            dmem[W1_BASE + g * L1_IN + i] = pack4_bf16(
                weights['w1'][4 * g + 3][i],
                weights['w1'][4 * g + 2][i],
                weights['w1'][4 * g + 1][i],
                weights['w1'][4 * g + 0][i],
            )

    # L1 biases: 4 packed words
    for g in range(L1_OUT // 4):
        dmem[B1_BASE + g] = pack4_bf16(
            weights['b1'][4 * g + 3],
            weights['b1'][4 * g + 2],
            weights['b1'][4 * g + 1],
            weights['b1'][4 * g + 0],
        )

    # L1 outputs [59..74] left zeroed (written by kernel)

    # L2 weights: 2 groups of 4 neurons, each with L2_IN input-words
    for g in range(L2_OUT // 4):
        for i in range(L2_IN):
            dmem[W2_BASE + g * L2_IN + i] = pack4_bf16(
                weights['w2'][4 * g + 3][i],
                weights['w2'][4 * g + 2][i],
                weights['w2'][4 * g + 1][i],
                weights['w2'][4 * g + 0][i],
            )

    # L2 biases: 2 packed words
    for g in range(L2_OUT // 4):
        dmem[B2_BASE + g] = pack4_bf16(
            weights['b2'][4 * g + 3],
            weights['b2'][4 * g + 2],
            weights['b2'][4 * g + 1],
            weights['b2'][4 * g + 0],
        )

    # L2 outputs [109..116] left zeroed (written by kernel)

    # L3 weights: 1 group of 2 neurons, packed as {0, 0, w[1][i], w[0][i]}
    for i in range(L3_IN):
        dmem[W3_BASE + i] = pack4_bf16(
            0.0, 0.0,
            weights['w3'][1][i],
            weights['w3'][0][i],
        )

    # L3 bias: {0, 0, b[1], b[0]}
    dmem[B3_BASE] = pack4_bf16(0.0, 0.0, weights['b3'][1], weights['b3'][0])

    # L3 output words [126..127] left zeroed (written by kernel)

    return dmem


def write_hex(dmem, path):
    with open(path, 'w') as f:
        for word in dmem:
            f.write(format_hex64(word) + '\n')


def main():
    output_path = 'data_ids_11_16_8_2_bcast.hex'
    use_simple = False
    weights_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '-o' and i + 1 < len(args):
            output_path = args[i + 1]; i += 2
        elif args[i] == '--test':
            use_simple = True; i += 1
        elif args[i] == '--weights' and i + 1 < len(args):
            weights_path = args[i + 1]; i += 2
        else:
            i += 1

    if weights_path:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        if 'weights' in weights and 'w1' in weights.get('weights', {}):
            weights = weights['weights']
        print("Loaded weights from %s" % weights_path)
    elif use_simple:
        weights = generate_simple_test_weights()
        print("Using simple test weights (all 0.25, biases 0)")
    else:
        weights = generate_test_weights()
        print("Using random test weights (seed=42)")

    inputs = [1.0] * L1_IN
    print("Test inputs: %s" % inputs)

    h1, h2, out = compute_reference(inputs, weights)
    print("\nReference (BF16-quantized):")
    print("  h1 = %s" % ['%.4f' % v for v in h1])
    print("  h2 = %s" % ['%.4f' % v for v in h2])
    print("  out = %s" % ['%.4f' % v for v in out])

    dmem = build_dmem(inputs, weights)
    write_hex(dmem, output_path)
    print("\nWrote %d DMEM words to %s" % (len(dmem), output_path))

    json_path = output_path.replace('.hex', '_weights.json')
    with open(json_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print("Saved weights to %s" % json_path)

    expected_path = output_path.replace('.hex', '_expected.txt')
    with open(expected_path, 'w') as f:
        f.write("# Expected DMEM values after BCAST kernel execution\n")
        f.write("# Format: DMEM_ADDR HEX_VALUE FLOAT_VALUE\n")
        for n in range(L1_OUT):
            word = replicate_bf16(h1[n])
            f.write("%d %s %.6f\n" % (H1_BASE + n, format_hex64(word), h1[n]))
        for n in range(L2_OUT):
            word = replicate_bf16(h2[n])
            f.write("%d %s %.6f\n" % (H2_BASE + n, format_hex64(word), h2[n]))
        for n in range(L3_OUT):
            word = replicate_bf16(out[n])
            f.write("%d %s %.6f\n" % (OUT_BASE + n, format_hex64(word), out[n]))
    print("Saved expected values to %s" % expected_path)


if __name__ == '__main__':
    main()
