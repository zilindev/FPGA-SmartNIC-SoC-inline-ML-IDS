#!/usr/bin/env python3
"""
bf16_utils.py — BFloat16 conversion utilities for custom FPGA GPU toolchain.
Matches the BF16 implementation in rtl/bf16_alu.v (truncation rounding).
"""

import struct


def float_to_bf16(f):
    """Convert Python float → 16-bit BF16 integer (truncation, matching HW).

    BF16 = upper 16 bits of IEEE 754 float32.
    """
    # Pack as float32, take upper 16 bits
    bits = struct.pack('>f', f)
    return (bits[0] << 8) | bits[1]


def bf16_to_float(b):
    """Convert 16-bit BF16 integer → Python float.

    BF16 is the upper 16 bits of float32; lower 16 bits are zero.
    """
    b = b & 0xFFFF
    # Expand to float32 by appending 16 zero bits
    raw = struct.pack('>HH', b, 0)
    return struct.unpack('>f', raw)[0]


def pack_bf16_vector(values):
    """Pack a list of 4 floats → single 64-bit int (4 × BF16 lanes).

    Lane layout: [63:48]=values[0], [47:32]=values[1], [31:16]=values[2], [15:0]=values[3]
    This matches the Verilog SIMD lane ordering.
    """
    assert len(values) == 4, f"Expected 4 values, got {len(values)}"
    result = 0
    for i, v in enumerate(values):
        bf = float_to_bf16(v)
        result |= bf << (48 - 16 * i)
    return result


def unpack_bf16_vector(word):
    """Unpack a 64-bit int → list of 4 Python floats (from BF16 lanes).

    Returns [lane3(MSB), lane2, lane1, lane0(LSB)] matching pack order.
    """
    word = word & 0xFFFFFFFFFFFFFFFF
    floats = []
    for i in range(4):
        lane = (word >> (48 - 16 * i)) & 0xFFFF
        floats.append(bf16_to_float(lane))
    return floats


def pack_int16_vector(values):
    """Pack a list of 4 int16 values → single 64-bit int.

    Lane layout: [63:48]=values[0], [47:32]=values[1], [31:16]=values[2], [15:0]=values[3]
    """
    assert len(values) == 4, f"Expected 4 values, got {len(values)}"
    result = 0
    for i, v in enumerate(values):
        # Convert to unsigned 16-bit representation
        v16 = v & 0xFFFF
        result |= v16 << (48 - 16 * i)
    return result


def unpack_int16_vector(word):
    """Unpack a 64-bit int → list of 4 signed int16 values."""
    word = word & 0xFFFFFFFFFFFFFFFF
    values = []
    for i in range(4):
        lane = (word >> (48 - 16 * i)) & 0xFFFF
        # Sign-extend
        if lane >= 0x8000:
            lane -= 0x10000
        values.append(lane)
    return values


def format_hex64(value):
    """Format a 64-bit value as a hex string for $readmemh (16 hex digits)."""
    return f"{value & 0xFFFFFFFFFFFFFFFF:016X}"


def format_hex32(value):
    """Format a 32-bit value as a hex string for $readmemh (8 hex digits)."""
    return f"{value & 0xFFFFFFFF:08X}"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("BF16 Utils Self-Test")
    print("=" * 60)

    # Known BF16 values from isa_encoding.md
    known = [
        (0.0,   0x0000),
        (0.25,  0x3E80),
        (0.5,   0x3F00),
        (1.0,   0x3F80),
        (2.0,   0x4000),
        (3.0,   0x4040),
        (6.0,   0x40C0),
        (-1.0,  0xBF80),
        (-2.0,  0xC000),
    ]

    all_pass = True
    for fval, expected_hex in known:
        got = float_to_bf16(fval)
        ok = got == expected_hex
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  float_to_bf16({fval:6.2f}) = 0x{got:04X}  (exp 0x{expected_hex:04X})")
        if not ok:
            all_pass = False

    print()

    # Round-trip test
    for fval, bf16_hex in known:
        rt = bf16_to_float(bf16_hex)
        ok = rt == fval
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  bf16_to_float(0x{bf16_hex:04X}) = {rt}  (exp {fval})")
        if not ok:
            all_pass = False

    print()

    # Pack/unpack test
    vec = [1.0, 2.0, 3.0, 0.5]
    packed = pack_bf16_vector(vec)
    print(f"  pack_bf16_vector({vec}) = 0x{packed:016X}")
    unpacked = unpack_bf16_vector(packed)
    ok = unpacked == vec
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  unpack round-trip: {unpacked}")
    if not ok:
        all_pass = False

    # Int16 pack/unpack
    ivec = [1, 2, 3, 4]
    ipacked = pack_int16_vector(ivec)
    expected_int = 0x0001000200030004
    ok = ipacked == expected_int
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  pack_int16_vector({ivec}) = 0x{ipacked:016X}  (exp 0x{expected_int:016X})")
    if not ok:
        all_pass = False

    # Signed int16
    svec = [-3, 5, -1, 7]
    spacked = pack_int16_vector(svec)
    expected_signed = 0xFFFD0005FFFF0007
    ok = spacked == expected_signed
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  pack_int16_vector({svec}) = 0x{spacked:016X}  (exp 0x{expected_signed:016X})")
    if not ok:
        all_pass = False

    sunpacked = unpack_int16_vector(spacked)
    ok = sunpacked == svec
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  unpack_int16_vector round-trip: {sunpacked}")
    if not ok:
        all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
