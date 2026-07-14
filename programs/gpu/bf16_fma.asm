# bf16_fma.asm — BFloat16 fused multiply-accumulate (4-lane SIMD, tensor core)
# DMEM[0] = A, DMEM[1] = B, DMEM[2] = C, DMEM[3] = result
# D[i] = A[i] * B[i] + C[i]
    LD   R1, 0(R0)          # R1 = A (4 x BF16)
    LD   R2, 1(R0)          # R2 = B (4 x BF16)
    LD   R3, 2(R0)          # R3 = C (4 x BF16)
    FMA  R4, R1, R2, R3     # R4 = A*B + C (per lane, BF16)
    ST   R4, 3(R0)          # DMEM[3] = result
    HALT
