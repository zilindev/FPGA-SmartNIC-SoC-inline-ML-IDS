# test_branch.asm — test label-based branching
    MOVI R1, 5
    MOVI R2, 10
    BLT  R1, R2, done   # R1 < R2 → branch to done
    MOVI R3, 0xFF       # should be skipped
done:
    HALT
