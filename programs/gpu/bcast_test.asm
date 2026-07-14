# BCAST instruction unit test
# DMEM[0] = {0xDDDD, 0xCCCC, 0xBBBB, 0xAAAA} (preloaded)
# Broadcasts each lane to all 4, stores results for verification

    LD    R1, 0(R0)       # R1 = DMEM[0] = {DDDD, CCCC, BBBB, AAAA}
    BCAST R2, R1, 0       # R2 = {AAAA, AAAA, AAAA, AAAA}
    BCAST R3, R1, 1       # R3 = {BBBB, BBBB, BBBB, BBBB}
    BCAST R4, R1, 2       # R4 = {CCCC, CCCC, CCCC, CCCC}
    BCAST R5, R1, 3       # R5 = {DDDD, DDDD, DDDD, DDDD}
    ST    R2, 1(R0)       # DMEM[1]
    ST    R3, 2(R0)       # DMEM[2]
    ST    R4, 3(R0)       # DMEM[3]
    ST    R5, 4(R0)       # DMEM[4]
    HALT
