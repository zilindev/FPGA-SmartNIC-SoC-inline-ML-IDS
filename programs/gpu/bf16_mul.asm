# bf16_mul.asm — BFloat16 vector multiply (4-lane SIMD)
# DMEM[0] = A, DMEM[1] = B, DMEM[2] = result
# C[i] = A[i] * B[i]
    LD   R1, 0(R0)              # R1 = A (4 x BF16)
    LD   R2, 1(R0)              # R2 = B (4 x BF16)
    MUL  R3, R1, R2, BF16      # R3 = A * B (per lane, BF16)
    ST   R3, 2(R0)              # DMEM[2] = result
    HALT
