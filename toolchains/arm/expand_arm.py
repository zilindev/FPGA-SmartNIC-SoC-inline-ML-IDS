#!/usr/bin/env python3
"""
expand_arm.py 

The output file is valid ARM assembly that can be compiled by
arm-none-eabi-as to produce machine code for the testbench.

Example Usage:
    python expand_arm.py sort.s
        -> creates sort_expanded.s (default reg-size=4)

    python expand_arm.py sort.s -o custom_name.s
        -> creates custom_name.s

    python expand_arm.py input.s --reg-size 8
        -> 64-bit mode (8 bytes per register slot in memory)

Supported expansions (shown with reg-size=4):
    push {fp, lr}       -> sub sp, sp, #8;  str fp,[sp,#0]; str lr,[sp,#4]
    pop  {fp, lr}       -> ldr fp,[sp,#0]; ldr lr,[sp,#4]; add sp, sp, #8
    ldmia Rn!, {r0-r3}  -> ldr r0,[Rn,#0]; ldr r1,[Rn,#4]; ...; add Rn, Rn, #16
    stmia Rn!, {r0-r3}  -> str r0,[Rn,#0]; str r1,[Rn,#4]; ...; add Rn, Rn, #16

All other lines pass through unchanged.
"""

import re
import sys
import os
import argparse

# ================================================================
# ARM register aliases
# ================================================================
REG_ALIASES = {
    'fp': 'r11', 'ip': 'r12', 'sp': 'r13', 'lr': 'r14', 'pc': 'r15'
}

def canonical_reg(name):
    """Convert register name to canonical form (e.g., 'fp' -> 'r11')."""
    name = name.strip().lower()
    return REG_ALIASES.get(name, name)

def reg_number(name):
    """Extract register number from canonical name."""
    name = canonical_reg(name)
    if name.startswith('r'):
        return int(name[1:])
    raise ValueError(f"Unknown register: {name}")

def reg_name(num):
    """Convert register number to display name, preserving aliases."""
    alias_map = {11: 'fp', 12: 'ip', 13: 'sp', 14: 'lr', 15: 'pc'}
    return alias_map.get(num, f'r{num}')

def parse_reglist(s):
    """Parse ARM register list like '{fp, lr}' or '{r0-r3, r5}'.
    Returns sorted list of register numbers (ARM always uses ascending order)."""
    s = s.strip().strip('{}')
    regs = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            lo_n = reg_number(lo)
            hi_n = reg_number(hi)
            for r in range(lo_n, hi_n + 1):
                regs.add(r)
        else:
            regs.add(reg_number(part))
    return sorted(regs)

# ================================================================
# Expansion functions
# ================================================================

def expand_push(reglist_str, indent, reg_size):
    """PUSH {reglist} = STMDB sp!, {reglist}
    Pre-decrement SP, then store lowest-numbered reg at lowest address."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    lines = []
    lines.append(f'{indent}@ --- PUSH {{{reglist_str}}} expanded ---')
    lines.append(f'{indent}sub\tsp, sp, #{total}')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}str\t{reg_name(r)}, [sp, #{offset}]')
    return lines

def expand_pop(reglist_str, indent, reg_size):
    """POP {reglist} = LDMIA sp!, {reglist}
    Load registers, then post-increment SP."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    lines = []
    lines.append(f'{indent}@ --- POP {{{reglist_str}}} expanded ---')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}ldr\t{reg_name(r)}, [sp, #{offset}]')
    lines.append(f'{indent}add\tsp, sp, #{total}')
    return lines

def expand_ldmia(base, reglist_str, writeback, indent, reg_size):
    """LDMIA Rn{!}, {reglist}
    Load registers from ascending addresses, optionally update base.
    If base is in reglist, writeback is skipped (loaded value wins)."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    base_num = reg_number(base)
    base_disp = reg_name(base_num)
    base_in_reglist = base_num in regs
    lines = []
    wb_str = '!' if writeback else ''
    lines.append(f'{indent}@ --- LDMIA {base}{wb_str}, {{{reglist_str}}} expanded ---')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}ldr\t{reg_name(r)}, [{base_disp}, #{offset}]')
    if writeback and not base_in_reglist:
        lines.append(f'{indent}add\t{base_disp}, {base_disp}, #{total}')
    elif writeback and base_in_reglist:
        lines.append(f'{indent}@ writeback skipped: {base_disp} in reglist (load wins)')
    return lines

def expand_stmia(base, reglist_str, writeback, indent, reg_size):
    """STMIA Rn{!}, {reglist}
    Store registers to ascending addresses, optionally update base."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    base_disp = reg_name(reg_number(base))
    lines = []
    wb_str = '!' if writeback else ''
    lines.append(f'{indent}@ --- STMIA {base}{wb_str}, {{{reglist_str}}} expanded ---')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}str\t{reg_name(r)}, [{base_disp}, #{offset}]')
    if writeback:
        lines.append(f'{indent}add\t{base_disp}, {base_disp}, #{total}')
    return lines

def expand_stmdb(base, reglist_str, writeback, indent, reg_size):
    """STMDB Rn!, {reglist}
    Pre-decrement base, then store lowest reg at lowest address."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    base_disp = reg_name(reg_number(base))
    lines = []
    wb_str = '!' if writeback else ''
    lines.append(f'{indent}@ --- STMDB {base}{wb_str}, {{{reglist_str}}} expanded ---')
    if writeback:
        lines.append(f'{indent}sub\t{base_disp}, {base_disp}, #{total}')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}str\t{reg_name(r)}, [{base_disp}, #{offset}]')
    return lines

def expand_ldmdb(base, reglist_str, writeback, indent, reg_size):
    """LDMDB Rn!, {reglist}
    Pre-decrement base, then load from ascending addresses."""
    regs = parse_reglist(reglist_str)
    total = len(regs) * reg_size
    base_disp = reg_name(reg_number(base))
    lines = []
    wb_str = '!' if writeback else ''
    lines.append(f'{indent}@ --- LDMDB {base}{wb_str}, {{{reglist_str}}} expanded ---')
    if writeback:
        lines.append(f'{indent}sub\t{base_disp}, {base_disp}, #{total}')
    for i, r in enumerate(regs):
        offset = i * reg_size
        lines.append(f'{indent}ldr\t{reg_name(r)}, [{base_disp}, #{offset}]')
    return lines

# ================================================================
# Pattern matching
# ================================================================

RE_PUSH  = re.compile(r'^(\s*)push\s+(\{[^}]+\})',                             re.I)
RE_POP   = re.compile(r'^(\s*)pop\s+(\{[^}]+\})',                              re.I)
RE_STMDB = re.compile(r'^(\s*)(?:stmdb|stmfd)\s+(\w+)(!)?\s*,\s*(\{[^}]+\})', re.I)
RE_LDMDB = re.compile(r'^(\s*)(?:ldmdb|ldmea)\s+(\w+)(!)?\s*,\s*(\{[^}]+\})', re.I)
RE_LDMIA = re.compile(r'^(\s*)(?:ldmia|ldm)\s+(\w+)(!)?\s*,\s*(\{[^}]+\})',   re.I)
RE_STMIA = re.compile(r'^(\s*)(?:stmia|stm)\s+(\w+)(!)?\s*,\s*(\{[^}]+\})',   re.I)

def expand_line(line, reg_size):
    """Expand one line. Returns list of output lines."""
    stripped = line.rstrip()

    m = RE_PUSH.match(stripped)
    if m:
        return expand_push(m.group(2).strip('{}'), m.group(1), reg_size)

    m = RE_POP.match(stripped)
    if m:
        return expand_pop(m.group(2).strip('{}'), m.group(1), reg_size)

    # Check STMDB/LDMDB before STMIA/LDMIA (longer prefix match first)
    m = RE_STMDB.match(stripped)
    if m:
        return expand_stmdb(m.group(2), m.group(4).strip('{}'),
                           m.group(3) == '!', m.group(1), reg_size)

    m = RE_LDMDB.match(stripped)
    if m:
        return expand_ldmdb(m.group(2), m.group(4).strip('{}'),
                           m.group(3) == '!', m.group(1), reg_size)

    m = RE_LDMIA.match(stripped)
    if m:
        return expand_ldmia(m.group(2), m.group(4).strip('{}'),
                           m.group(3) == '!', m.group(1), reg_size)

    m = RE_STMIA.match(stripped)
    if m:
        return expand_stmia(m.group(2), m.group(4).strip('{}'),
                           m.group(3) == '!', m.group(1), reg_size)

    return [stripped]

def expand_file(input_lines, reg_size=4):
    """Process all lines. Returns (output_lines, stats_dict)."""
    output = []
    stats = {'push': 0, 'pop': 0, 'ldm': 0, 'stm': 0}

    for line in input_lines:
        original = line.rstrip()
        expanded = expand_line(original, reg_size)

        if len(expanded) > 1:
            lower = original.strip().lower()
            if lower.startswith('push'):    stats['push'] += 1
            elif lower.startswith('pop'):   stats['pop']  += 1
            elif 'ldm' in lower[:6]:        stats['ldm']  += 1
            elif 'stm' in lower[:6]:        stats['stm']  += 1

        output.extend(expanded)

    return output, stats

# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Expand ARM multi-register instructions for EE 533 Lab 6')
    parser.add_argument('input', help='Input ARM assembly file (.s)')
    parser.add_argument('-o', '--output', help='Output file (default: <input>_expanded.s)')
    parser.add_argument('--reg-size', type=int, default=4,
                       help='Bytes per register in memory (default: 4 for standard ARM)')
    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out_path = f'{base}_expanded{ext}'

    with open(args.input, 'r') as f:
        input_lines = f.readlines()

    output_lines, stats = expand_file(input_lines, args.reg_size)
    result = '\n'.join(output_lines) + '\n'

    with open(out_path, 'w') as f:
        f.write(result)

    # Print summary
    total = sum(stats.values())
    print(f'Expanded {total} instructions (PUSH:{stats["push"]} POP:{stats["pop"]} '
          f'LDM:{stats["ldm"]} STM:{stats["stm"]})')
    print(f'  reg-size = {args.reg_size} bytes')
    print(f'  Output:  {out_path}')
    print()
    #print('Next steps:')
    #print(f'  1. Assemble:  arm-none-eabi-as -march=armv4t -o sort.o {out_path}')
    #print(f'  2. Link:      arm-none-eabi-ld -Ttext=0x0 -o sort.elf sort.o')
    #print(f'  3. Dump:      arm-none-eabi-objdump -d sort.elf')
    #print(f'  4. Provide the hex machine code to generate the Verilog testbench')

if __name__ == '__main__':
    main()
