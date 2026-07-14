#!/usr/bin/env python3
"""
dump2hex.py — Convert ARM objdump disassembly to a plain .hex file.

Parses sort_exp_dump.txt (UTF-16 encoded) and extracts machine code words.
Output: one 32-bit hex word per line (no 0x prefix), suitable for processorreg.

Usage:
    python dump2hex.py sort_exp_dump.txt -o sort.hex
    python dump2hex.py sort_exp_dump.txt              # default: sort_exp_dump.hex
"""

import re
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Convert ARM objdump to plain .hex file')
    parser.add_argument('input', help='Objdump disassembly file')
    parser.add_argument('-o', '--output', help='Output .hex file')
    args = parser.parse_args()

    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(args.input)
        out_path = base + '.hex'

    # Try UTF-16 first (Windows objdump), fall back to UTF-8
    for enc in ('utf-16', 'utf-8', 'latin-1'):
        try:
            with open(args.input, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print("ERROR: Cannot decode input file.", file=sys.stderr)
        sys.exit(1)

    # Match objdump lines like:
    #    0:	e24dd008 	sub	sp, sp, #8
    # or data words like:
    #  170:	00000000 	.word	0x00000000
    pattern = re.compile(r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\b')

    words = []  # (byte_addr, hex_encoding)
    for line in lines:
        m = pattern.match(line)
        if m:
            byte_addr = int(m.group(1), 16)
            encoding = m.group(2).lower()
            words.append((byte_addr, encoding))

    if not words:
        print("ERROR: No machine code found in input.", file=sys.stderr)
        sys.exit(1)

    # Verify words are contiguous starting at 0
    for i, (addr, _) in enumerate(words):
        expected = i * 4
        if addr != expected:
            print(f"WARNING: Gap at word {i}: expected addr 0x{expected:x}, "
                  f"got 0x{addr:x}", file=sys.stderr)

    with open(out_path, 'w') as f:
        f.write(f"// {len(words)} instructions from {os.path.basename(args.input)}\n")
        f.write(f"// Addresses 0x000 .. 0x{words[-1][0]:03x} "
                f"({len(words)} words)\n")
        for addr, enc in words:
            f.write(f"{enc}\n")

    print(f"Wrote {len(words)} words to {out_path}")
    print(f"  Address range: 0x{words[0][0]:03x} .. 0x{words[-1][0]:03x}")


if __name__ == '__main__':
    main()
