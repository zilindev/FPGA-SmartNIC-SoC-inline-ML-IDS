# ann_ids_11_16_8_2.asm — IDS MLP inference kernel (11 -> 16 -> 8 -> 2)
#
# Network: 3-layer MLP for network intrusion detection
#   Layer 1: 11 inputs  -> 16 neurons, ReLU activation
#   Layer 2: 16 inputs  ->  8 neurons, ReLU activation
#   Layer 3:  8 inputs  ->  2 outputs, no activation (raw logits)
#
# Execution mode: SCALAR (all 4 SIMD lanes replicated with same value)
#   - Enables correct inter-layer data flow without BCAST instruction
#   - All DMEM values stored as {v, v, v, v} across 4 BF16 lanes
#   - Pointer increment via ADD Rd, Rs, R7 (not ADDI) for lane uniformity
#
# Optimization: 2-neuron interleaving per inner loop
#   - Hides LD->FMA pipeline stall latency (~14 cycles/iter for 2 FMAs)
#   - ~2470 cycles total, ~20 us at 125 MHz
#
# Register allocation (inner loop):
#   R0  = zero (hardwired)          R7  = {1,1,1,1} increment constant
#   R1  = accumulator neuron A      R8  = weight pointer
#   R2  = weight temp A             R9  = input pointer / loop counter
#   R3  = input temp (shared)       R10 = (free for setup/teardown)
#   R4  = weight temp B             R11 = (free for setup/teardown)
#   R5  = accumulator neuron B      R12 = loop limit
#   R6  = (free for setup/teardown)
#
# DMEM layout (383 words, all values replicated across 4 BF16 lanes):
#   [0..10]     Input features x[0..10]                    (11 words)
#   [11..186]   L1 weights, neuron-major: w1[n][i] @ 11+n*11+i  (176 words)
#   [187..202]  L1 biases b1[0..15]                        (16 words)
#   [203..218]  L1 output h1[0..15] = L2 input             (16 words)
#   [219..346]  L2 weights: w2[n][i] @ 219+n*16+i          (128 words)
#   [347..354]  L2 biases b2[0..7]                         (8 words)
#   [355..362]  L2 output h2[0..7] = L3 input              (8 words)
#   [363..378]  L3 weights: w3[n][i] @ 363+n*8+i           (16 words)
#   [379..380]  L3 biases b3[0..1]                         (2 words)
#   [381..382]  L3 output out[0..1]                        (2 words)

# =====================================================================
# Global setup
# =====================================================================
    MOVI R7, 1               # R7 = {1,1,1,1} for uniform pointer increment

# =====================================================================
# Layer 1: 11 -> 16, ReLU  (8 groups × 2 neurons, stride = 11)
# =====================================================================
    MOVI R12, 11             # loop limit = 0 + 11

    # --- L1 Group 0: neurons 0, 1 ---
    LD   R1, 187(R0)         # acc_A = b1[0]
    LD   R5, 188(R0)         # acc_B = b1[1]
    MOVI R8, 11              # weight ptr -> w1[0][0]
    MOVI R9, 0               # input ptr -> x[0]
l1g0:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)          # stride = 11
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g0
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 203(R0)         # h1[0]
    ST   R5, 204(R0)         # h1[1]

    # --- L1 Group 1: neurons 2, 3 ---
    LD   R1, 189(R0)
    LD   R5, 190(R0)
    MOVI R8, 33              # 11 + 2*11
    MOVI R9, 0
l1g1:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g1
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 205(R0)
    ST   R5, 206(R0)

    # --- L1 Group 2: neurons 4, 5 ---
    LD   R1, 191(R0)
    LD   R5, 192(R0)
    MOVI R8, 55              # 11 + 4*11
    MOVI R9, 0
l1g2:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g2
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 207(R0)
    ST   R5, 208(R0)

    # --- L1 Group 3: neurons 6, 7 ---
    LD   R1, 193(R0)
    LD   R5, 194(R0)
    MOVI R8, 77              # 11 + 6*11
    MOVI R9, 0
l1g3:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g3
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 209(R0)
    ST   R5, 210(R0)

    # --- L1 Group 4: neurons 8, 9 ---
    LD   R1, 195(R0)
    LD   R5, 196(R0)
    MOVI R8, 99              # 11 + 8*11
    MOVI R9, 0
l1g4:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g4
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 211(R0)
    ST   R5, 212(R0)

    # --- L1 Group 5: neurons 10, 11 ---
    LD   R1, 197(R0)
    LD   R5, 198(R0)
    MOVI R8, 121             # 11 + 10*11
    MOVI R9, 0
l1g5:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g5
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 213(R0)
    ST   R5, 214(R0)

    # --- L1 Group 6: neurons 12, 13 ---
    LD   R1, 199(R0)
    LD   R5, 200(R0)
    MOVI R8, 143             # 11 + 12*11
    MOVI R9, 0
l1g6:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g6
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 215(R0)
    ST   R5, 216(R0)

    # --- L1 Group 7: neurons 14, 15 ---
    LD   R1, 201(R0)
    LD   R5, 202(R0)
    MOVI R8, 165             # 11 + 14*11
    MOVI R9, 0
l1g7:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1g7
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 217(R0)
    ST   R5, 218(R0)

# =====================================================================
# Layer 2: 16 -> 8, ReLU  (4 groups × 2 neurons, stride = 16)
# =====================================================================
    MOVI R12, 219            # loop limit = 203 + 16

    # --- L2 Group 0: neurons 0, 1 ---
    LD   R1, 347(R0)
    LD   R5, 348(R0)
    MOVI R8, 219             # W2_BASE
    MOVI R9, 203             # H1_BASE
l2g0:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 16(R8)          # stride = 16
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l2g0
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 355(R0)
    ST   R5, 356(R0)

    # --- L2 Group 1: neurons 2, 3 ---
    LD   R1, 349(R0)
    LD   R5, 350(R0)
    MOVI R8, 251             # 219 + 2*16
    MOVI R9, 203
l2g1:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 16(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l2g1
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 357(R0)
    ST   R5, 358(R0)

    # --- L2 Group 2: neurons 4, 5 ---
    LD   R1, 351(R0)
    LD   R5, 352(R0)
    MOVI R8, 283             # 219 + 4*16
    MOVI R9, 203
l2g2:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 16(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l2g2
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 359(R0)
    ST   R5, 360(R0)

    # --- L2 Group 3: neurons 6, 7 ---
    LD   R1, 353(R0)
    LD   R5, 354(R0)
    MOVI R8, 315             # 219 + 6*16
    MOVI R9, 203
l2g3:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 16(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l2g3
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    ST   R1, 361(R0)
    ST   R5, 362(R0)

# =====================================================================
# Layer 3: 8 -> 2, no activation  (1 group × 2 neurons, stride = 8)
# =====================================================================
    MOVI R12, 363            # loop limit = 355 + 8

    # --- L3 Group 0: neurons 0, 1 ---
    LD   R1, 379(R0)
    LD   R5, 380(R0)
    MOVI R8, 363             # W3_BASE
    MOVI R9, 355             # H2_BASE
l3g0:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 8(R8)           # stride = 8
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l3g0

    # No activation on output layer (raw logits for classification)
    ST   R1, 381(R0)         # out[0]
    ST   R5, 382(R0)         # out[1]
    HALT
