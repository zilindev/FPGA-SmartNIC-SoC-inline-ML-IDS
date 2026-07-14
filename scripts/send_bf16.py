#!/usr/bin/env python
# -*- coding: ascii -*-
#
# send_bf16.py -- Send UDP packet with BF16 data to NetFPGA FIFO
#
# Sends a UDP datagram with 6 bytes of padding + 8 bytes of BF16 data.
# The padding aligns the BF16 data to BRAM[7] (byte offset 48) in the
# NetFPGA FIFO, so the FPGA can DMA it directly.
#
# Usage:
#   python send_bf16.py                   # default: {1.0, 2.0, 3.0, 4.0}
#   python send_bf16.py 3F804000_3F004040 # custom 8 bytes as hex
#
# Compatible with Python 2.4+ and Python 3.x.

import socket
import sys

FPGA_IP = '10.0.4.255'    # Broadcast on 10.0.4.0/24 (nf6 nf2c0)
FPGA_PORT = 9999

# 6 bytes padding aligns BF16 data to 8-byte BRAM word boundary
# Packet layout in FIFO BRAM:
#   BRAM[0]:   NF2.1 header (ctrl=0xFF)
#   BRAM[1-6]: Eth + IP + UDP headers + padding
#   BRAM[7]:   our 8 bytes of BF16 data  <-- DMA from here
PADDING = b'\x00\x00\x00\x00\x00\x00'

# Default: {1.0, 2.0, 3.0, 4.0} as 4x BF16
# BF16: 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040, 4.0=0x4080
DEFAULT_HEX = '3F80400040404080'

# BF16 value table for reference
BF16_TABLE = {
    0x0000: '0', 0x3F00: '0.5', 0x3F80: '1.0', 0x3FC0: '1.5',
    0x4000: '2.0', 0x4020: '2.5', 0x4040: '3.0', 0x4060: '3.5',
    0x4080: '4.0', 0x40A0: '5.0', 0x40C0: '6.0', 0x40E0: '7.0',
    0x4100: '8.0', 0x4110: '9.0', 0x4120: '10.0', 0x40B0: '5.5',
    0x4108: '8.5', 0x4138: '11.5', 0x3E80: '0.25', 0x3FE0: '1.75',
    0x4098: '4.75', 0x40F8: '7.75',
}


def hex_to_bytes(hex_str):
    """Convert hex string to bytes. Python 2.4 compatible."""
    hex_str = hex_str.replace('_', '').replace(' ', '')
    result = b''
    for i in range(0, len(hex_str), 2):
        result += bytes(bytearray([int(hex_str[i:i+2], 16)]))
    return result


def bf16_name(val):
    """Look up BF16 value in table, or return hex."""
    if val in BF16_TABLE:
        return BF16_TABLE[val]
    return '0x%04X' % val


def main():
    if len(sys.argv) > 1:
        hex_str = sys.argv[1]
    else:
        hex_str = DEFAULT_HEX

    data = hex_to_bytes(hex_str)
    if len(data) != 8:
        print('ERROR: need exactly 8 bytes (16 hex chars), got %d' % len(data))
        sys.exit(1)

    payload = PADDING + data

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(payload, (FPGA_IP, FPGA_PORT))
    s.close()

    # Decode and display
    vals = []
    for i in range(0, 8, 2):
        if sys.version_info[0] >= 3:
            v = data[i] * 256 + data[i+1]
        else:
            v = ord(data[i]) * 256 + ord(data[i+1])
        vals.append(v)

    print('Sent UDP to %s:%d' % (FPGA_IP, FPGA_PORT))
    print('  BF16 data: [%s]' % ', '.join(['0x%04X' % v for v in vals]))
    print('  As floats: [%s]' % ', '.join([bf16_name(v) for v in vals]))
    print('  Hex:       0x%04X%04X_%04X%04X' % (vals[0], vals[1], vals[2], vals[3]))
    print('  Lands at BRAM[7] in NetFPGA FIFO')


if __name__ == '__main__':
    main()
