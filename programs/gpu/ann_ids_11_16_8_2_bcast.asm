# ann_ids_11_16_8_2_bcast.asm — BCAST-optimized IDS inference kernel
#
# Network: 3-layer MLP for intrusion detection (11 -> 16 -> 8 -> 2).
#
# Execution mode: 4-neuron SIMD parallel via BCAST.
#   Weights are packed 4-per-word: {w[4g+3][i], w[4g+2][i], w[4g+1][i], w[4g+0][i]}
#   Inputs are scalar-replicated {x,x,x,x}, so a single FMA per cycle
#   updates 4 independent neuron accumulators in one register:
#       FMA R1, R2, R3, R1  ->  R1[j] += w[j]*x  for j=0..3
#
# Super-group structure: 2-way register interleave (R1+R5) to hide LD->FMA
# pipeline stalls, matching the proven scalar kernel pattern.
#   Layer 1: 2 super-groups x 8 neurons each  (16 neurons)
#   Layer 2: 1 super-group  x 8 neurons       (8 neurons)
#   Layer 3: 1 group, 2-neuron packed mode    (2 outputs; lanes 2,3 zeroed)
#
# BCAST unpack: after each layer, the 4-lane packed accumulator is split
# into 4 scalar (replicated) words via BCAST Rd, Rs, {0,1,2,3} + ST, so the
# next layer can treat its inputs as {v,v,v,v}.
#
# DMEM layout (128 words):
#   [0..10]     Input x[0..10]                       (scalar, 11 words)
#   [11..54]    L1 weights packed 4-per-word          (44 words)
#   [55..58]    L1 biases packed 4-per-word           ( 4 words)
#   [59..74]    L1 outputs h1[0..15] (scalar)         (16 words, written)
#   [75..106]   L2 weights packed 4-per-word          (32 words)
#   [107..108]  L2 biases packed 4-per-word           ( 2 words)
#   [109..116]  L2 outputs h2[0..7] (scalar)          ( 8 words, written)
#   [117..124]  L3 weights packed {0,0,w1[i],w0[i]}   ( 8 words)
#   [125]       L3 biases packed {0,0,b1,b0}          ( 1 word)
#   [126..127]  L3 outputs out[0..1] (scalar)         ( 2 words, written)
#
# Register allocation:
#   R0  = zero (hardwired)               R7  = {1,1,1,1} pointer increment
#   R1  = accumulator neurons A          R8  = weight pointer
#   R5  = accumulator neurons B          R9  = input pointer / loop counter
#   R2  = weight temp A / BCAST temp     R11 = BCAST unpack temp
#   R4  = weight temp B / BCAST temp     R12 = loop limit
#   R3  = input temp / BCAST temp        R6, R10 = free
#
# NOTE: R13, R14, R15 are hardwired read-only (THREADID, BLOCKID, BLOCKDIM);
# they cannot be used as BCAST destinations. R2, R3, R4 are reused as BCAST
# temps after each FMA loop completes (they are dead at that point).

# =====================================================================
# Global setup
# =====================================================================
    MOVI R7, 1                  # {1,1,1,1} pointer increment

# =====================================================================
# Layer 1: 11 inputs -> 16 neurons, ReLU  (2 super-groups x 8 neurons)
# Weights are laid out as groups of 4 packed neurons; stride between
# adjacent groups is L1_IN = 11 words.
# =====================================================================
    MOVI R12, 11                # inner loop limit = 11 (input count)

    # --- L1 Super-group 0: neurons 0..7 ---
    LD   R1, 55(R0)             # R1 = {b1[3], b1[2], b1[1], b1[0]}
    LD   R5, 56(R0)             # R5 = {b1[7], b1[6], b1[5], b1[4]}
    MOVI R8, 11                 # W1 group-0 base (neurons 0-3)
    MOVI R9, 0                  # input base
l1sg0:
    LD   R2, 0(R8)              # packed weights group 0
    LD   R3, 0(R9)              # scalar input x[i]
    ADD  R9, R9, R7
    LD   R4, 11(R8)             # packed weights group 1 (stride = 11)
    FMA  R1, R2, R3, R1         # neurons 0..3 accumulate
    FMA  R5, R4, R3, R5         # neurons 4..7 accumulate
    ADD  R8, R8, R7
    BLT  R9, R12, l1sg0
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    # Unpack R1 -> h1[0..3] scalar
    BCAST R11, R1, 0
    BCAST R2, R1, 1
    BCAST R3, R1, 2
    BCAST R4, R1, 3
    ST   R11, 59(R0)            # h1[0]
    ST   R2, 60(R0)            # h1[1]
    ST   R3, 61(R0)            # h1[2]
    ST   R4, 62(R0)            # h1[3]
    # Unpack R5 -> h1[4..7] scalar
    BCAST R11, R5, 0
    BCAST R2, R5, 1
    BCAST R3, R5, 2
    BCAST R4, R5, 3
    ST   R11, 63(R0)            # h1[4]
    ST   R2, 64(R0)            # h1[5]
    ST   R3, 65(R0)            # h1[6]
    ST   R4, 66(R0)            # h1[7]

    # --- L1 Super-group 1: neurons 8..15 ---
    LD   R1, 57(R0)             # R1 = {b1[11],b1[10],b1[9], b1[8]}
    LD   R5, 58(R0)             # R5 = {b1[15],b1[14],b1[13],b1[12]}
    MOVI R8, 33                 # W1 group-2 base = 11 + 2*11
    MOVI R9, 0
l1sg1:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 11(R8)
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l1sg1
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    # Unpack R1 -> h1[8..11] scalar
    BCAST R11, R1, 0
    BCAST R2, R1, 1
    BCAST R3, R1, 2
    BCAST R4, R1, 3
    ST   R11, 67(R0)            # h1[8]
    ST   R2, 68(R0)            # h1[9]
    ST   R3, 69(R0)            # h1[10]
    ST   R4, 70(R0)            # h1[11]
    # Unpack R5 -> h1[12..15] scalar
    BCAST R11, R5, 0
    BCAST R2, R5, 1
    BCAST R3, R5, 2
    BCAST R4, R5, 3
    ST   R11, 71(R0)            # h1[12]
    ST   R2, 72(R0)            # h1[13]
    ST   R3, 73(R0)            # h1[14]
    ST   R4, 74(R0)            # h1[15]

# =====================================================================
# Layer 2: 16 inputs -> 8 neurons, ReLU  (1 super-group x 8 neurons)
# Inputs at [59..74] are scalar (BCAST-unpacked from L1).
# Weight stride between groups is L2_IN = 16.
# =====================================================================
    MOVI R12, 75                # loop limit = H1_BASE + L2_IN = 59 + 16

    LD   R1, 107(R0)            # R1 = {b2[3], b2[2], b2[1], b2[0]}
    LD   R5, 108(R0)            # R5 = {b2[7], b2[6], b2[5], b2[4]}
    MOVI R8, 75                 # W2 group-0 base
    MOVI R9, 59                 # H1 base (scalar)
l2sg0:
    LD   R2, 0(R8)
    LD   R3, 0(R9)
    ADD  R9, R9, R7
    LD   R4, 16(R8)             # stride = L2_IN = 16
    FMA  R1, R2, R3, R1
    FMA  R5, R4, R3, R5
    ADD  R8, R8, R7
    BLT  R9, R12, l2sg0
    RELU R1, R1, BF16
    RELU R5, R5, BF16
    # Unpack R1 -> h2[0..3] scalar
    BCAST R11, R1, 0
    BCAST R2, R1, 1
    BCAST R3, R1, 2
    BCAST R4, R1, 3
    ST   R11, 109(R0)           # h2[0]
    ST   R2, 110(R0)           # h2[1]
    ST   R3, 111(R0)           # h2[2]
    ST   R4, 112(R0)           # h2[3]
    # Unpack R5 -> h2[4..7] scalar
    BCAST R11, R5, 0
    BCAST R2, R5, 1
    BCAST R3, R5, 2
    BCAST R4, R5, 3
    ST   R11, 113(R0)           # h2[4]
    ST   R2, 114(R0)           # h2[5]
    ST   R3, 115(R0)           # h2[6]
    ST   R4, 116(R0)           # h2[7]

# =====================================================================
# Layer 3: 8 inputs -> 2 outputs, no activation (raw logits for softmax)
# 2-neuron packed mode: weights are {0, 0, w3[1][i], w3[0][i]} so only
# lanes 0,1 carry the real computation; lanes 2,3 compute 0*x+0 = 0.
# =====================================================================
    MOVI R12, 117               # loop limit = H2_BASE + L3_IN = 109 + 8

    LD   R1, 125(R0)            # R1 = {0, 0, b3[1], b3[0]}
    MOVI R8, 117                # W3 base
    MOVI R9, 109                # H2 base (scalar)
l3g0:
    LD   R2, 0(R8)              # packed weights
    LD   R3, 0(R9)              # scalar input
    ADD  R9, R9, R7
    ADD  R8, R8, R7
    FMA  R1, R2, R3, R1
    BLT  R9, R12, l3g0

    # No activation. Unpack lanes 0,1 to scalar out[0], out[1].
    BCAST R11, R1, 0
    BCAST R2, R1, 1
    ST   R11, 126(R0)           # out[0]
    ST   R2, 127(R0)           # out[1]
    HALT
