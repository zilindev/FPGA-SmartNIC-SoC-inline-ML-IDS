# test_fma.asm — matches kernel_bf16_fma_tb.v FMA program
# Expected: 80800000 / 81000001 / 81800002 / 22094300 / 8A000003 / F8000000
    LD   R1, 0(R0)          # load A (BF16)
    LD   R2, 1(R0)          # load B (BF16)
    LD   R3, 2(R0)          # load C (BF16)
    FMA  R4, R1, R2, R3     # D = A*B + C (BF16)
    ST   R4, 3(R0)          # store result
    HALT
