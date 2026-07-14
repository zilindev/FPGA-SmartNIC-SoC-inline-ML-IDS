# test_vec_add.asm — matches kernel_ldst_tb.v vec_add program
# Expected: 80800000 / 81000001 / 09890000 / 89800002 / F8000000
    LD   R1, 0(R0)      # load A
    LD   R2, 1(R0)      # load B
    ADD  R3, R1, R2     # C = A + B
    ST   R3, 2(R0)      # store result
    HALT
