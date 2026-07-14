# vec_add_i16.asm — Vector addition (int16, 4-lane SIMD)
# DMEM[0] = A, DMEM[1] = B, DMEM[2] = result
# C[i] = A[i] + B[i]
    LD   R1, 0(R0)      # R1 = A (4 x int16)
    LD   R2, 1(R0)      # R2 = B (4 x int16)
    ADD  R3, R1, R2     # R3 = A + B (per lane)
    ST   R3, 2(R0)      # DMEM[2] = result
    HALT
