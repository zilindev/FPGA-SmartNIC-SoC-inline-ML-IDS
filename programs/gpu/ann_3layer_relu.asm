# ann_inference.asm — 3-layer scalar ReLU network, batch-4 BF16 inference
#
# Network: f(x) = 3.0 * ReLU(0.5 * ReLU(2.0 * x - 1.0) + 0.25) - 0.5
# Weights: w1=2.0  b1=-1.0  w2=0.5  b2=0.25  w3=3.0  b3=-0.5
#
# DMEM layout:
#   [0] = x       4 input values (one per BF16 SIMD lane)
#   [1] = w1      broadcast 2.0
#   [2] = b1      broadcast -1.0
#   [3] = w2      broadcast 0.5
#   [4] = b2      broadcast 0.25
#   [5] = w3      broadcast 3.0
#   [6] = b3      broadcast -0.5
#   [7] = output  4 results

    # Layer 1: h1 = ReLU(w1 * x + b1)
    LD   R1, 0(R0)              # R1 = inputs
    LD   R2, 1(R0)              # R2 = w1 (broadcast 2.0)
    LD   R3, 2(R0)              # R3 = b1 (broadcast -1.0)
    FMA  R4, R2, R1, R3         # R4 = 2.0*x + (-1.0) = 2x - 1
    RELU R4, R4                 # R4 = ReLU(2x - 1)

    # Layer 2: h2 = ReLU(w2 * h1 + b2)
    LD   R5, 3(R0)              # R5 = w2 (broadcast 0.5)
    LD   R6, 4(R0)              # R6 = b2 (broadcast 0.25)
    FMA  R7, R5, R4, R6         # R7 = 0.5*h1 + 0.25
    RELU R7, R7                 # R7 = ReLU(0.5*h1 + 0.25)

    # Layer 3: out = w3 * h2 + b3
    LD   R8, 5(R0)              # R8 = w3 (broadcast 3.0)
    LD   R9, 6(R0)              # R9 = b3 (broadcast -0.5)
    FMA  R10, R8, R7, R9        # R10 = 3.0*h2 + (-0.5)

    # Store result
    ST   R10, 7(R0)             # DMEM[7] = output
    HALT
