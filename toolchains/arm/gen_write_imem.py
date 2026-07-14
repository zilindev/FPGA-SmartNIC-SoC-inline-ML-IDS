#!/usr/bin/env python3
"""
Parse an ARM objdump disassembly and produce write_imem() calls
for a Verilog testbench.

Usage:
    python3 gen_write_imem.py <objdump_file> [--start-word <N>]

Output goes to stdout — redirect or copy-paste into processor_tb.v.
"""

import re
import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <objdump_file> [--start-word <N>]", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    start_word = 0
    if '--start-word' in sys.argv:
        idx = sys.argv.index('--start-word')
        start_word = int(sys.argv[idx + 1])

    # Regex to match objdump disassembly lines like:
    #    0:	e24dd008 	sub	sp, sp, #8
    # or data words like:
    #  170:	00000000 	.word	0x00000000
    pattern = re.compile(
        r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s+(.*)$'
    )

    entries = []
    with open(filename, 'r', encoding='utf-16') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                byte_addr = int(m.group(1), 16)
                encoding  = m.group(2)
                comment   = m.group(3).strip()
                word_addr = byte_addr // 4
                entries.append((byte_addr, word_addr, encoding, comment))

    if not entries:
        print("ERROR: No instructions found in the input file.", file=sys.stderr)
        sys.exit(1)

    print(f"        // ========================================================")
    print(f"        // Auto-generated from: {filename}")
    print(f"        // {len(entries)} words, IMEM words {start_word}..{start_word + len(entries) - 1}")
    print(f"        // ========================================================")

    for byte_addr, word_addr, encoding, comment in entries:
        imem_word = start_word + word_addr
        # Truncate very long comments
        if len(comment) > 60:
            comment = comment[:57] + "..."
        print(f"        write_imem(9'd{imem_word:<3d}, 32'h{encoding.upper()}); "
              f"// @0x{byte_addr:03X}: {comment}")

    print()
    print(f"        // Total: {len(entries)} words loaded into IMEM[{start_word}..{start_word + len(entries) - 1}]")

if __name__ == '__main__':
    main()
