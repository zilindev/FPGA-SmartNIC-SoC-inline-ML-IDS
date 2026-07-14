# relu_i16.asm — ReLU activation (int16, 4-lane SIMD)
# DMEM[0] = input, DMEM[1] = result
# out[i] = max(0, in[i])
    LD   R1, 0(R0)      # R1 = input (4 x int16)
    RELU R2, R1         # R2 = max(0, R1) per lane
    ST   R2, 1(R0)      # DMEM[1] = result
    HALT
