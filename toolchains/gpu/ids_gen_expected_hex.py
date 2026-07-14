#!/usr/bin/env python3
"""Convert expected.txt from ids_data_generator.py to $readmemh hex file.

Extracts 26 expected DMEM values (16 L1 + 8 L2 + 2 L3) in order.

Usage: python ids_gen_expected_hex.py expected.txt -o expected.hex
"""
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python ids_gen_expected_hex.py expected.txt [-o output.hex]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = "expected_ids.hex"
    if '-o' in sys.argv:
        output_path = sys.argv[sys.argv.index('-o') + 1]

    with open(input_path, 'r') as f:
        lines = f.readlines()

    values = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        values.append(parts[1])  # hex value

    with open(output_path, 'w') as f:
        for v in values:
            f.write(v + '\n')

    print(f"Wrote {len(values)} expected values to {output_path}")

if __name__ == '__main__':
    main()
