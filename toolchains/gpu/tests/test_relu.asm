# test_relu.asm — matches kernel_ldst_tb.v relu program
# Expected: 80800000 / C1080000 / 89000001 / F8000000
    LD   R1, 0(R0)      # load input
    RELU R2, R1         # R2 = max(0, R1) per lane
    ST   R2, 1(R0)      # store result
    HALT
