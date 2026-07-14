#!/usr/bin/env python3
"""
train_ann.py -- Train a 3-layer scalar ReLU network and export BF16 weights
for deployment on the FPGA GPU.

Network: x -> ReLU(w1*x + b1) -> ReLU(w2*h1 + b2) -> w3*h2 + b3 -> output
Target function: f(x) = clamp(x^2 / 4, 0, 2)  (nonlinear, bounded)

Training with numpy, manual backprop. Exports:
  - BF16 weight hex values for DMEM
  - Test vectors with expected outputs
  - Assembly program (same as Option A, weights differ)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Lab7_synth', 'toolchain'))
from bf16_utils import float_to_bf16, bf16_to_float, pack_bf16_vector, format_hex64


# ============================================================================
# BF16 quantization helpers
# ============================================================================

def to_bf16(x):
    """Round-trip float through BF16."""
    return bf16_to_float(float_to_bf16(float(x)))

def quantize_weights(w1, b1, w2, b2, w3, b3):
    """Quantize all weights to BF16."""
    return (to_bf16(w1), to_bf16(b1),
            to_bf16(w2), to_bf16(b2),
            to_bf16(w3), to_bf16(b3))


# ============================================================================
# Network: forward and backward (float32 training)
# ============================================================================

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = w1 * x + b1
    h1 = relu(z1)
    z2 = w2 * h1 + b2
    h2 = relu(z2)
    out = w3 * h2 + b3
    return out, (z1, h1, z2, h2)

def backward(x, y_true, w1, b1, w2, b2, w3, b3):
    out, (z1, h1, z2, h2) = forward(x, w1, b1, w2, b2, w3, b3)
    loss = np.mean((out - y_true) ** 2)

    # d_loss/d_out = 2 * (out - y_true) / N
    N = len(x)
    d_out = 2.0 * (out - y_true) / N

    # Layer 3: out = w3 * h2 + b3
    dw3 = np.sum(d_out * h2)
    db3 = np.sum(d_out)
    d_h2 = d_out * w3

    # Layer 2: h2 = ReLU(w2 * h1 + b2)
    d_z2 = d_h2 * relu_grad(z2)
    dw2 = np.sum(d_z2 * h1)
    db2 = np.sum(d_z2)
    d_h1 = d_z2 * w2

    # Layer 1: h1 = ReLU(w1 * x + b1)
    d_z1 = d_h1 * relu_grad(z1)
    dw1 = np.sum(d_z1 * x)
    db1 = np.sum(d_z1)

    return loss, (dw1, db1, dw2, db2, dw3, db3)


# ============================================================================
# Target function
# ============================================================================

def target_fn(x):
    """f(x) = clamp(2*x - 1, 0, 3) -- ramp with saturation.
    Piecewise linear: 0 for x<=0.5, 2x-1 for 0.5<x<2, 3 for x>=2.
    A 3-layer scalar ReLU network can learn this exactly."""
    return np.clip(2.0 * x - 1.0, 0.0, 3.0)


# ============================================================================
# Training
# ============================================================================

def train():
    np.random.seed(42)

    # Training data: uniform samples in [-1, 4]
    x_train = np.linspace(-1, 4, 300)
    y_train = target_fn(x_train)

    # Initialize weights (positive to avoid dead ReLU)
    w1 = 0.5 + np.random.rand() * 0.5
    b1 = -0.3 + np.random.randn() * 0.1
    w2 = -0.5 + np.random.randn() * 0.3
    b2 = 0.5 + np.random.randn() * 0.1
    w3 = 0.5 + np.random.rand() * 0.5
    b3 = np.random.randn() * 0.1

    lr = 0.005
    print("Training 3-layer scalar ReLU network...")
    print("Target: f(x) = clamp(2x - 1, 0, 3)")
    print("")

    for epoch in range(5000):
        loss, (dw1, db1, dw2, db2, dw3, db3) = backward(
            x_train, y_train, w1, b1, w2, b2, w3, b3)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        w3 -= lr * dw3
        b3 -= lr * db3

        if epoch % 1000 == 0:
            print("  Epoch %4d  loss=%.6f" % (epoch, loss))

    print("  Epoch %4d  loss=%.6f (final)" % (epoch, loss))
    print("")

    # Print learned weights
    print("Learned weights (float32):")
    print("  w1=%8.4f  b1=%8.4f" % (w1, b1))
    print("  w2=%8.4f  b2=%8.4f" % (w2, b2))
    print("  w3=%8.4f  b3=%8.4f" % (w3, b3))
    print("")

    # Quantize to BF16
    w1q, b1q, w2q, b2q, w3q, b3q = quantize_weights(w1, b1, w2, b2, w3, b3)
    print("Quantized weights (BF16):")
    print("  w1=%8.4f (0x%04X)  b1=%8.4f (0x%04X)" % (w1q, float_to_bf16(w1q), b1q, float_to_bf16(b1q)))
    print("  w2=%8.4f (0x%04X)  b2=%8.4f (0x%04X)" % (w2q, float_to_bf16(w2q), b2q, float_to_bf16(b2q)))
    print("  w3=%8.4f (0x%04X)  b3=%8.4f (0x%04X)" % (w3q, float_to_bf16(w3q), b3q, float_to_bf16(b3q)))
    print("")

    return w1q, b1q, w2q, b2q, w3q, b3q


# ============================================================================
# Generate test vectors and DMEM data
# ============================================================================

def generate_test(w1, b1, w2, b2, w3, b3):
    # Test inputs: 4 values spanning all 3 regions (flat/ramp/saturated)
    test_inputs = [-0.5, 0.5, 1.5, 3.0]

    print("Test vectors (BF16 inference):")
    print("  %-8s  %-10s  %-10s  %-10s  %-10s" % ("x", "target", "h1", "h2", "output"))

    results = []
    for x in test_inputs:
        xb = to_bf16(x)
        h1 = to_bf16(w1 * xb + b1)
        h1 = to_bf16(max(0.0, h1))
        h2 = to_bf16(w2 * h1 + b2)
        h2 = to_bf16(max(0.0, h2))
        out = to_bf16(w3 * h2 + b3)
        tgt = target_fn(np.array([xb]))[0]
        results.append(out)
        print("  %-8.2f  %-10.4f  %-10.4f  %-10.4f  %-10.4f" % (xb, tgt, h1, h2, out))

    print("")

    # Pack DMEM words
    dmem = [
        pack_bf16_vector(test_inputs),      # DMEM[0] = inputs
        pack_bf16_vector([w1] * 4),         # DMEM[1] = w1 broadcast
        pack_bf16_vector([b1] * 4),         # DMEM[2] = b1 broadcast
        pack_bf16_vector([w2] * 4),         # DMEM[3] = w2 broadcast
        pack_bf16_vector([b2] * 4),         # DMEM[4] = b2 broadcast
        pack_bf16_vector([w3] * 4),         # DMEM[5] = w3 broadcast
        pack_bf16_vector([b3] * 4),         # DMEM[6] = b3 broadcast
    ]
    expected = pack_bf16_vector(results)    # DMEM[7] = expected output

    print("DMEM data for gpureg.py:")
    labels = ["x (inputs)", "w1 broadcast", "b1 broadcast",
              "w2 broadcast", "b2 broadcast", "w3 broadcast", "b3 broadcast"]
    for i, (d, label) in enumerate(zip(dmem, labels)):
        h = format_hex64(d)
        print("  dmem_wr %d %s %s    # %s" % (i, h[:8], h[8:], label))
    h = format_hex64(expected)
    print("")
    print("Expected DMEM[7] = 0x%s_%s" % (h[:8], h[8:]))

    # Print per-lane breakdown
    print("")
    print("Per-lane expected:")
    for i, (x, r) in enumerate(zip(test_inputs, results)):
        print("  input=%.2f  output=%.4f  BF16=0x%04X" % (x, r, float_to_bf16(r)))

    return test_inputs, results, dmem, expected


# ============================================================================
# Export for gpureg.py test
# ============================================================================

def export_test_data(test_inputs, results, dmem, expected):
    """Print Python code snippet for gpureg.py test function."""
    print("")
    print("=" * 60)
    print("Add this to gpureg.py as test_ann_trained():")
    print("=" * 60)
    print("")
    print("def test_ann_trained():")
    print('    return run_and_check("ANN trained: f(x)=clamp(2x-1,0,3) (BF16)", "ann_inference.hex",')
    print("        [", end="")
    labels = ["x", "w1", "b1", "w2", "b2", "w3", "b3"]
    for i, (d, label) in enumerate(zip(dmem, labels)):
        h = format_hex64(d)
        hi = int(h[:8], 16)
        lo = int(h[8:], 16)
        if i > 0:
            print("         ", end="")
        print("(%d, 0x%08X, 0x%08X)," % (i, hi, lo), end="")
        print("   # %s" % label)
    h = format_hex64(expected)
    hi = int(h[:8], 16)
    lo = int(h[8:], 16)
    print("        7, 0x%08X, 0x%08X)" % (hi, lo))
    print("")


# ============================================================================
# Main
# ============================================================================

def main():
    w1, b1, w2, b2, w3, b3 = train()
    test_inputs, results, dmem, expected = generate_test(w1, b1, w2, b2, w3, b3)
    export_test_data(test_inputs, results, dmem, expected)


if __name__ == '__main__':
    main()
