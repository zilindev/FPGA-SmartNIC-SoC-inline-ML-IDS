# test_integer.asm — matches programs/test_integer.hex
# Expected: D8800005 / D9000003 / 09890000 / F8000000
    MOVI R1, 5
    MOVI R2, 3
    ADD  R3, R1, R2
    HALT
