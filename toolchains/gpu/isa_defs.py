#!/usr/bin/env python3
"""
isa_defs.py — ISA definitions for custom FPGA GPU processor
Matches gpu_params.vh exactly. Single source of truth for the Python toolchain.
"""

# ---------------------------------------------------------------------------
# Instruction field positions (bit ranges within 32-bit instruction)
# ---------------------------------------------------------------------------
INSTR_WIDTH = 32

# Common fields
F_OPCODE = (31, 27)   # 5 bits
F_RD     = (26, 23)   # 4 bits
F_RS1    = (22, 19)   # 4 bits
F_RS2    = (18, 15)   # 4 bits
F_DTYPE  = (14, 12)   # 3 bits
F_FUNC   = (11, 0)    # 12 bits

# I-type: imm19 in [18:0]
F_IMM19  = (18, 0)    # 19 bits

# M-type: offset12 in [11:0]
F_OFFSET12 = (11, 0)  # 12 bits

# F-type (FMA): rs3 in [11:8]
F_RS3    = (11, 8)     # 4 bits

# B-type: offset19 in [18:0] (same position as IMM19)
F_BOFFSET = (18, 0)   # 19 bits

# MOVI: imm16 in [15:0] (broadcast to all 4 SIMD lanes by hardware)
F_MOVI_IMM = (15, 0)  # 16 bits

# ---------------------------------------------------------------------------
# Opcode encoding (5 bits) — must match gpu_params.vh
# ---------------------------------------------------------------------------
OPCODES = {
    'NOP':  0b00000,  # 0x00
    'ADD':  0b00001,  # 0x01
    'SUB':  0b00010,  # 0x02
    'MUL':  0b00011,  # 0x03
    'FMA':  0b00100,  # 0x04
    'MAX':  0b00101,  # 0x05
    'MOV':  0b01100,  # 0x0C
    'ADDI': 0b01110,  # 0x0E
    'LD':   0b10000,  # 0x10
    'ST':   0b10001,  # 0x11
    'BLT':  0b10110,  # 0x16
    'BGE':  0b10111,  # 0x17
    'RELU': 0b11000,  # 0x18
    'BCAST': 0b11010, # 0x1A — Cross-lane broadcast
    'MOVI': 0b11011,  # 0x1B
    'HALT': 0b11111,  # 0x1F
}

# ---------------------------------------------------------------------------
# Data type encoding (3 bits) — dtype field [14:12]
# ---------------------------------------------------------------------------
DTYPES = {
    'S16':  0b000,
    'U16':  0b001,
    'S32':  0b010,
    'BF16': 0b100,
}

# Default dtype per opcode (used when dtype not specified in assembly)
DEFAULT_DTYPE = {
    'ADD': 'S16', 'SUB': 'S16', 'MUL': 'S16', 'MAX': 'S16',
    'MOV': 'S16', 'RELU': 'S16', 'FMA': 'BF16',
}

# ---------------------------------------------------------------------------
# Register names → numbers
# ---------------------------------------------------------------------------
REGISTERS = {}
for i in range(16):
    REGISTERS[f'R{i}'] = i

# Aliases
REGISTERS['ZERO']     = 0
REGISTERS['THREADID'] = 13
REGISTERS['BLOCKID']  = 14
REGISTERS['BLOCKDIM'] = 15

# ---------------------------------------------------------------------------
# Instruction format classification
# ---------------------------------------------------------------------------
# R-type: ADD, SUB, MUL, MAX, MOV, RELU
# I-type: MOVI, ADDI
# M-type: LD, ST
# F-type: FMA
# B-type: BLT, BGE
# Special: NOP, HALT

FORMAT_R = {'ADD', 'SUB', 'MUL', 'MAX', 'MOV', 'RELU', 'BCAST'}
FORMAT_I = {'MOVI', 'ADDI'}
FORMAT_M = {'LD', 'ST'}
FORMAT_F = {'FMA'}
FORMAT_B = {'BLT', 'BGE'}
FORMAT_SPECIAL = {'NOP', 'HALT'}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _field(value, hi, lo):
    """Place value into bit field [hi:lo] of a 32-bit word."""
    width = hi - lo + 1
    mask = (1 << width) - 1
    return (value & mask) << lo


def encode_r_type(opcode, rd, rs1, rs2, dtype_str='S16'):
    """R-type: opcode rd, rs1, rs2 [, dtype]"""
    return (_field(OPCODES[opcode], 31, 27) |
            _field(rd, 26, 23) |
            _field(rs1, 22, 19) |
            _field(rs2, 18, 15) |
            _field(DTYPES[dtype_str], 14, 12))


def encode_relu(rd, rs1, dtype_str='S16'):
    """RELU rd, rs1 [, dtype] — R-type with rs2=0"""
    return encode_r_type('RELU', rd, rs1, 0, dtype_str)


def encode_mov(rd, rs1, dtype_str='S16'):
    """MOV rd, rs1 [, dtype] — R-type with rs2=0"""
    return encode_r_type('MOV', rd, rs1, 0, dtype_str)


def encode_movi(rd, imm16):
    """MOVI rd, imm16 — I-type with 16-bit immediate in [15:0]"""
    return (_field(OPCODES['MOVI'], 31, 27) |
            _field(rd, 26, 23) |
            _field(imm16, 15, 0))


def encode_addi(rd, rs1, imm19):
    """ADDI rd, rs1, imm19 — I-type"""
    return (_field(OPCODES['ADDI'], 31, 27) |
            _field(rd, 26, 23) |
            _field(rs1, 22, 19) |
            _field(imm19, 18, 0))


def encode_m_type(opcode, rd, rbase, offset12, dtype_str='S16'):
    """M-type: LD/ST rd, offset(rbase) [, dtype]"""
    return (_field(OPCODES[opcode], 31, 27) |
            _field(rd, 26, 23) |
            _field(rbase, 22, 19) |
            _field(DTYPES[dtype_str], 14, 12) |
            _field(offset12, 11, 0))


def encode_fma(rd, rs1, rs2, rs3, dtype_str='BF16'):
    """F-type: FMA rd, rs1, rs2, rs3 [, dtype]"""
    return (_field(OPCODES['FMA'], 31, 27) |
            _field(rd, 26, 23) |
            _field(rs1, 22, 19) |
            _field(rs2, 18, 15) |
            _field(DTYPES[dtype_str], 14, 12) |
            _field(rs3, 11, 8))


def encode_branch(opcode, rs1, rs2, offset19):
    """B-type: BLT/BGE rs1, rs2, offset"""
    return (_field(OPCODES[opcode], 31, 27) |
            _field(rs1, 26, 23) |
            _field(rs2, 22, 19) |
            _field(offset19, 18, 0))


def encode_bcast(rd, rs1, lane):
    """BCAST rd, rs1, lane — cross-lane broadcast (lane in func[1:0])"""
    return (_field(OPCODES['BCAST'], 31, 27) |
            _field(rd, 26, 23) |
            _field(rs1, 22, 19) |
            _field(lane & 0x3, 1, 0))


def encode_nop():
    """NOP — all zeros"""
    return 0x00000000


def encode_halt():
    """HALT"""
    return _field(OPCODES['HALT'], 31, 27)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Verify against hand-assembled known-good encodings from isa_encoding.md
    tests = [
        ('MOVI R1, 5',          encode_movi(1, 5),                  0xD8800005),
        ('MOVI R2, 3',          encode_movi(2, 3),                  0xD9000003),
        ('ADD R3,R1,R2 S16',    encode_r_type('ADD', 3, 1, 2),     0x09890000),
        ('SUB R3,R1,R2 S16',    encode_r_type('SUB', 3, 1, 2),     0x11890000),
        ('MUL R3,R1,R2 BF16',   encode_r_type('MUL', 3, 1, 2, 'BF16'), 0x19894000),
        ('LD R1, 0(R0)',        encode_m_type('LD', 1, 0, 0),      0x80800000),
        ('LD R2, 1(R0)',        encode_m_type('LD', 2, 0, 1),      0x81000001),
        ('ST R3, 2(R0)',        encode_m_type('ST', 3, 0, 2),      0x89800002),
        ('ST R2, 1(R0)',        encode_m_type('ST', 2, 0, 1),      0x89000001),
        ('RELU R2, R1',         encode_relu(2, 1),                  0xC1080000),
        ('FMA R4,R1,R2,R3 BF16', encode_fma(4, 1, 2, 3),          0x22094300),
        ('HALT',                encode_halt(),                      0xF8000000),
        ('NOP',                 encode_nop(),                       0x00000000),
        ('BCAST R3,R1,0',      encode_bcast(3, 1, 0),              0xD1880000),
        ('BCAST R3,R1,2',      encode_bcast(3, 1, 2),              0xD1880002),
    ]

    print("ISA Encoding Self-Test")
    print("=" * 60)
    all_pass = True
    for name, got, expected in tests:
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:<30s}  got={got:08X}  exp={expected:08X}")
        if not ok:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print(f"  ALL {len(tests)} TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
