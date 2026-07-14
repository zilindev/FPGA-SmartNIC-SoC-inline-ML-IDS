#!/usr/bin/env python3
"""
assembler.py — 2-pass assembler for custom FPGA GPU ISA.
Reads .asm text, outputs $readmemh-compatible .hex (one 32-bit word per line).

Usage:
    python assembler.py input.asm [-o output.hex]

Assembly syntax:
    MOVI R1, 0x0005          # I-type: load immediate
    ADDI R3, R1, 10          # I-type: add immediate
    ADD  R3, R1, R2          # R-type: default S16
    ADD  R3, R1, R2, BF16    # R-type: explicit dtype
    MUL  R3, R1, R2, BF16    # R-type with dtype
    RELU R2, R1              # R-type: rd, rs1
    MOV  R2, R1              # R-type: rd, rs1
    FMA  R4, R1, R2, R3      # F-type: rd, rs1, rs2, rs3 (default BF16)
    LD   R1, 0(R0)           # M-type: rd, offset(rbase)
    ST   R1, 2(R0)           # M-type: rd, offset(rbase)
    BLT  R1, R2, label       # B-type: branch if rs1 < rs2
    BGE  R1, R2, label       # B-type: branch if rs1 >= rs2
    NOP                      # pseudo: all zeros
    HALT                     # pseudo: halt execution
    label:                   # label definition (for branches)

Comments: # or // to end of line
"""

import sys
import re
import os
from isa_defs import (
    OPCODES, DTYPES, REGISTERS, DEFAULT_DTYPE,
    FORMAT_R, FORMAT_I, FORMAT_M, FORMAT_F, FORMAT_B, FORMAT_SPECIAL,
    encode_r_type, encode_relu, encode_mov, encode_movi, encode_addi,
    encode_m_type, encode_fma, encode_branch, encode_nop, encode_halt,
    encode_bcast,
)


def parse_register(token):
    """Parse a register name (R0-R15 or aliases) → register number."""
    token = token.strip().rstrip(',').upper()
    if token in REGISTERS:
        return REGISTERS[token]
    raise ValueError(f"Unknown register: '{token}'")


def parse_immediate(token):
    """Parse an immediate value (decimal, hex with 0x prefix, or negative)."""
    token = token.strip().rstrip(',')
    if token.startswith('0x') or token.startswith('0X'):
        return int(token, 16)
    if token.startswith('-0x') or token.startswith('-0X'):
        return -int(token[1:], 16)
    return int(token)


def parse_mem_operand(token):
    """Parse memory operand 'offset(rbase)' → (offset, rbase_num)."""
    token = token.strip()
    m = re.match(r'(-?\w+)\((\w+)\)', token)
    if not m:
        raise ValueError(f"Invalid memory operand: '{token}'")
    offset = parse_immediate(m.group(1))
    rbase = parse_register(m.group(2))
    return offset, rbase


def strip_comment(line):
    """Remove comments (# or //) from a line."""
    # Handle // first, then #
    for marker in ['//', '#']:
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


class Assembler:
    def __init__(self):
        self.labels = {}       # label_name → PC address
        self.instructions = [] # list of (line_num, raw_line, tokens)

    def pass1(self, lines):
        """Pass 1: collect labels and strip them from instruction lines."""
        pc = 0
        self.instructions = []

        for line_num, raw_line in enumerate(lines, 1):
            line = strip_comment(raw_line)
            if not line:
                continue

            # Check for label definition: "label_name:"
            if ':' in line:
                parts = line.split(':', 1)
                label = parts[0].strip()
                if label:
                    self.labels[label] = pc
                # Anything after the colon is an instruction on the same line
                line = parts[1].strip()
                if not line:
                    continue

            # Tokenize
            # Split by commas and whitespace, but keep memory operands intact
            tokens = self._tokenize(line)
            if tokens:
                self.instructions.append((line_num, raw_line.strip(), tokens, pc))
                pc += 1

    def _tokenize(self, line):
        """Split instruction line into tokens, handling memory operands."""
        # Replace commas with spaces, but preserve offset(reg) patterns
        # First, find and protect memory operands
        protected = line
        mem_ops = re.findall(r'-?\w+\(\w+\)', protected)

        # Simple approach: split by comma and whitespace
        parts = re.split(r'[,\s]+', line)
        parts = [p for p in parts if p]
        return parts

    def pass2(self):
        """Pass 2: encode each instruction into a 32-bit word."""
        encoded = []
        for line_num, raw, tokens, pc in self.instructions:
            try:
                word = self._encode(tokens, pc)
                encoded.append(word)
            except Exception as e:
                raise ValueError(f"Line {line_num}: {raw}\n  Error: {e}")
        return encoded

    def _encode(self, tokens, pc):
        """Encode a single instruction from tokens."""
        mnemonic = tokens[0].upper()
        args = tokens[1:]

        if mnemonic == 'NOP':
            return encode_nop()

        if mnemonic == 'HALT':
            return encode_halt()

        if mnemonic == 'MOVI':
            # MOVI rd, imm16
            rd = parse_register(args[0])
            imm = parse_immediate(args[1])
            return encode_movi(rd, imm)

        if mnemonic == 'ADDI':
            # ADDI rd, rs1, imm19
            rd = parse_register(args[0])
            rs1 = parse_register(args[1])
            imm = parse_immediate(args[2])
            return encode_addi(rd, rs1, imm)

        if mnemonic in FORMAT_R:
            if mnemonic == 'RELU':
                # RELU rd, rs1 [, dtype]
                rd = parse_register(args[0])
                rs1 = parse_register(args[1])
                dtype = args[2].upper() if len(args) > 2 else DEFAULT_DTYPE.get(mnemonic, 'S16')
                return encode_relu(rd, rs1, dtype)

            if mnemonic == 'MOV':
                # MOV rd, rs1 [, dtype]
                rd = parse_register(args[0])
                rs1 = parse_register(args[1])
                dtype = args[2].upper() if len(args) > 2 else DEFAULT_DTYPE.get(mnemonic, 'S16')
                return encode_mov(rd, rs1, dtype)

            if mnemonic == 'BCAST':
                # BCAST rd, rs1, lane (lane is 0-3 immediate)
                rd = parse_register(args[0])
                rs1 = parse_register(args[1])
                lane = parse_immediate(args[2])
                return encode_bcast(rd, rs1, lane)

            # ADD, SUB, MUL, MAX: rd, rs1, rs2 [, dtype]
            rd = parse_register(args[0])
            rs1 = parse_register(args[1])
            rs2 = parse_register(args[2])
            dtype = args[3].upper() if len(args) > 3 else DEFAULT_DTYPE.get(mnemonic, 'S16')
            return encode_r_type(mnemonic, rd, rs1, rs2, dtype)

        if mnemonic in FORMAT_M:
            # LD/ST rd, offset(rbase) [, dtype]
            rd = parse_register(args[0])
            offset, rbase = parse_mem_operand(args[1])
            dtype = args[2].upper() if len(args) > 2 else 'S16'
            return encode_m_type(mnemonic, rd, rbase, offset, dtype)

        if mnemonic == 'FMA':
            # FMA rd, rs1, rs2, rs3 [, dtype]
            rd = parse_register(args[0])
            rs1 = parse_register(args[1])
            rs2 = parse_register(args[2])
            rs3 = parse_register(args[3])
            dtype = args[4].upper() if len(args) > 4 else DEFAULT_DTYPE.get('FMA', 'BF16')
            return encode_fma(rd, rs1, rs2, rs3, dtype)

        if mnemonic in FORMAT_B:
            # BLT/BGE rs1, rs2, label_or_offset
            rs1 = parse_register(args[0])
            rs2 = parse_register(args[1])
            target = args[2].strip()

            if target in self.labels:
                # PC-relative offset (in words)
                offset = self.labels[target] - pc
            else:
                # Numeric offset
                offset = parse_immediate(target)

            return encode_branch(mnemonic, rs1, rs2, offset)

        raise ValueError(f"Unknown mnemonic: '{mnemonic}'")


def assemble_file(input_path, output_path=None):
    """Assemble a .asm file → .hex file."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    asm = Assembler()
    asm.pass1(lines)
    encoded = asm.pass2()

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + '.hex'

    with open(output_path, 'w') as f:
        for word in encoded:
            f.write(f"{word:08X}\n")

    return encoded, output_path


def assemble_string(text):
    """Assemble from a string → list of 32-bit encoded words."""
    lines = text.strip().split('\n')
    asm = Assembler()
    asm.pass1(lines)
    return asm.pass2()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python assembler.py input.asm [-o output.hex]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        output_path = sys.argv[idx + 1]

    encoded, out = assemble_file(input_path, output_path)
    print(f"Assembled {len(encoded)} instructions -> {out}")
    for i, word in enumerate(encoded):
        print(f"  [{i:3d}] {word:08X}")
