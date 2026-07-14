#!/usr/bin/env python
# -*- coding: ascii -*-
#
# send_ids.py -- Send IDS feature vectors as UDP packets to NetFPGA
#
# Sends 11 BF16 feature values to the NetFPGA FIFO via UDP.
# Uses the same 6-byte padding trick as send_bf16.py to align data
# to BRAM[7] in the NetFPGA FIFO.
#
# Packet layouts (selected by --replicated flag):
#
#   Packed (default; matches lab10reg.py ids_batch_packets PCI flow):
#     BRAM[0]:     NF2.1 header (ctrl=0xFF)
#     BRAM[1-6]:   Eth + IP + UDP headers (42 bytes + 6 padding = 48 bytes)
#     BRAM[7]:     Features x[0..3]  (4 x BF16 = 8 bytes)
#     BRAM[8]:     Features x[4..7]  (4 x BF16 = 8 bytes)
#     BRAM[9]:     Features x[8..10] (3 x BF16 = 6 bytes, 2 bytes padding)
#
#   Replicated (--replicated; matches ARM orchestrator's direct DMA path):
#     BRAM[0]:     NF2.1 header
#     BRAM[1-6]:   Eth + IP + UDP headers + 6 padding
#     BRAM[7..17]: one BF16 feature per word, replicated 4x across SIMD lanes
#                  (matches data_ids_11_16_8_2_bcast.hex DMEM[0..10] convention)
#     See docs/arm_orchestrator.md section 13 for why this split exists.
#
# Usage:
#   python send_ids.py                              # default attack sample, packed
#   python send_ids.py --replicated                 # default sample, replicated
#   python send_ids.py 0.1 0.2 0.3 ... (11 floats)  # normalized features, packed
#   python send_ids.py --csv features.csv            # batch from CSV, packed
#   python send_ids.py --csv features.csv --replicated
#   python send_ids.py --raw 0 0 0.04 ... --norm     # raw + auto-normalize
#
# Compatible with Python 2.4+ and Python 3.x (for NetFPGA node0).

import socket
import select
import sys
import os
import struct
import time

FPGA_IP = '10.0.7.255'    # Broadcast on 10.0.4.0/22 (nf7 port0 subnet, bypasses ARP)
FPGA_PORT = 9999

# Local UDP port used when --listen is set. Bound on this side before send,
# then sendto() uses it as the UDP source port. The FPGA response swaps
# UDP src/dst so responses arrive back at this port on the same socket.
LISTEN_PORT = 9998

# 6 bytes padding aligns BF16 data to BRAM[7]
PADDING = '\x00\x00\x00\x00\x00\x00'

N_FEATURES = 11

# Default test sample: row 2 of v3 test_features.csv (attack, true_label=1).
# Feature order matches the v3 trained model's selected_features:
#   dst_bytes, src_bytes, flag, same_srv_rate, diff_srv_rate,
#   dst_host_srv_count, dst_host_same_srv_rate, logged_in, protocol_type,
#   count, service.
# Sending v1-ordered features here would mis-feed the v3 model.
DEFAULT_FEATURES = [0.000000, 0.000000, 0.200000, 0.020000, 0.070000,
                    0.198120, 0.010000, 0.000000, 0.000000, 0.756373,
                    0.028986]


def float_to_bf16(f):
    """Convert float to 16-bit BF16 (truncation, matching hardware)."""
    raw = struct.pack('>f', f)
    return (ord(raw[0]) << 8) | ord(raw[1])


def bf16_to_float(b):
    """Convert 16-bit BF16 to float."""
    b = b & 0xFFFF
    raw = struct.pack('>HH', b, 0)
    return struct.unpack('>f', raw)[0]


def hex_to_bytes_compat(hex_str):
    """Convert hex string to bytes. Python 2.4 compatible."""
    hex_str = hex_str.replace('_', '').replace(' ', '')
    result = ''
    for i in range(0, len(hex_str), 2):
        result += chr(int(hex_str[i:i + 2], 16))
    return result


def pack_features(features):
    """Pack 11 float features as BF16 bytes (22 bytes, padded to 32).

    Packed layout: 4 BF16 per 8-byte word. Padding to 32 bytes ensures
    total frame = 42+6+32 = 80 bytes, giving word_length=10 in NF2.1
    header, so FIFO stores 9 data words (BRAM[1..9] = 72 bytes) which
    covers features at BRAM[7..9]. This is what lab10reg.py's PCI flow
    `ids_batch_packets` expects.
    """
    data = ''
    for f in features:
        bf = float_to_bf16(f)
        data += struct.pack('>H', bf)
    # Pad from 22 to 32 bytes (5 zero BF16 values)
    while len(data) < 32:
        data += struct.pack('>H', 0x0000)
    return data


def pack_features_replicated(features):
    """Pack 11 float features as 11 replicated 8-byte words + 8-byte trailer.

    Replicated layout: each 8-byte word is one BF16 value repeated 4x
    across the 4 SIMD lanes (bits [63:48, 47:32, 31:16, 15:0]). Lands
    in the FIFO at BRAM[7..17] after the 6-byte payload padding, which
    matches the convention the BCAST IDS kernel expects at GPU DMEM[0..10]
    (see data_ids_11_16_8_2_bcast.hex lines 1..11: all 3F80_3F80_3F80_3F80).

    Trailer-word rationale (FIFO FSM word_length mismatch):
      The convertible_fifo FSM at src/fifo/convertible_fifo.v:164 treats
      the NF2 header's word_length as "total NF2 words including header"
      and initializes words_remaining = pkt_len - 1. The real NF2 classifier
      on silicon instead reports word_length = data-word count, so the FSM
      captures one fewer data word than arrived -- the LAST word lands in
      BRAM[tail] but never commits. Without a trailer, BRAM[17] (the 11th
      feature) would be silently dropped and the BCAST kernel would read
      lane-0 of DMEM[10] as 0 (off-by-2.0 in the output). Adding one 8-byte
      zero trailer makes the frame 144 bytes = 18 data words so the classifier
      reports 18, FSM captures 17 data words, and BRAM[1..17] are all valid.
      This is a silicon-only workaround; the sim TB injects via fifo_write
      directly and uses its own word_length, so it is unaffected.
      See docs/arm_orchestrator.md section 13 for the diagnostic narrative.

    Total frame = 42 + 6 + 88 + 8 = 144 bytes -> 17 data words captured by
    the FIFO FSM (plus the NF2 header), covering BRAM[0..17]. Consumed by
    the ARM orchestrator's 11-word DMA from FIFO[7..17] -> DMEM[0..10].
    """
    data = ''
    for f in features:
        bf = float_to_bf16(f)
        # Replicate BF16 across all 4 lanes of one 8-byte (64-bit) word.
        # Big-endian packing: H H H H gives word[63:48|47:32|31:16|15:0]
        # all equal to bf. Lane-0 BCAST of this in the BCAST kernel picks
        # up `bf` correctly regardless of which lane it BCASTs from.
        data += struct.pack('>HHHH', bf, bf, bf, bf)
    # 8-byte zero trailer to work around the FIFO FSM off-by-one truncation
    # (see docstring above). Lands in BRAM[18] and gets silently dropped,
    # which is the intended sacrificial word.
    data += '\x00\x00\x00\x00\x00\x00\x00\x00'
    return data


def send_features(features, verbose=True, replicated=False):
    """Send 11 features as a UDP packet to the NetFPGA."""
    if len(features) != N_FEATURES:
        print("ERROR: expected %d features, got %d" % (N_FEATURES, len(features)))
        return False

    if replicated:
        data = pack_features_replicated(features)  # 88 bytes
    else:
        data = pack_features(features)             # 32 bytes (22 data + 10 pad)
    payload = PADDING + data

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(payload, (FPGA_IP, FPGA_PORT))
    s.close()

    if verbose:
        bfs = []
        for f in features:
            bf = float_to_bf16(f)
            bfs.append(bf)

        if replicated:
            layout = "replicated (BRAM[7..17])"
        else:
            layout = "packed (BRAM[7..9])"
        print("Sent %d features to %s:%d  [%s]" %
              (N_FEATURES, FPGA_IP, FPGA_PORT, layout))
        for i in range(N_FEATURES):
            print("  x[%2d] = %8.4f -> BF16 0x%04X" % (i, features[i], bfs[i]))
        if replicated:
            for i in range(N_FEATURES):
                print("  BRAM[%d]: 0x%04X%04X%04X%04X" %
                      (7 + i, bfs[i], bfs[i], bfs[i], bfs[i]))
        else:
            print("  BRAM[7]: 0x%04X%04X%04X%04X" % (bfs[0], bfs[1], bfs[2], bfs[3]))
            print("  BRAM[8]: 0x%04X%04X%04X%04X" % (bfs[4], bfs[5], bfs[6], bfs[7]))
            print("  BRAM[9]: 0x%04X%04X%04X0000" % (bfs[8], bfs[9], bfs[10]))

    return True


def load_normalization_params():
    """Load normalization parameters from trained weights JSON."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'programs', 'gpu',
                             'data_ids_trained_weights.json')
    if not os.path.exists(json_path):
        return None

    try:
        import json
        f = open(json_path, 'r')
        data = json.load(f)
        f.close()
        return data.get('norm_params', None)
    except Exception:
        return None


def normalize_features(raw_features, norm_params):
    """Normalize raw features to [0, 1] using training min/max."""
    mins = norm_params['mins']
    maxs = norm_params['maxs']
    normalized = []
    for i in range(len(raw_features)):
        r = maxs[i] - mins[i]
        if r == 0:
            r = 1.0
        val = (raw_features[i] - mins[i]) / r
        val = max(0.0, min(1.0, val))  # clamp
        normalized.append(val)
    return normalized


def load_samples(csv_path, do_normalize):
    """Parse a CSV into a list of normalized [0..1] feature lists."""
    norm_params = None
    if do_normalize:
        norm_params = load_normalization_params()
        if norm_params is None:
            print("ERROR: cannot load normalization params")
            return [], None

    f = open(csv_path, 'r')
    lines = f.readlines()
    f.close()

    start = 0
    try:
        float(lines[0].strip().split(',')[0])
    except (ValueError, IndexError):
        start = 1

    samples = []
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < N_FEATURES:
            continue
        features = [float(p.strip()) for p in parts[:N_FEATURES]]
        if do_normalize and norm_params:
            features = normalize_features(features, norm_params)
        samples.append(features)

    return samples, lines


def send_batch(csv_path, delay=0.5, do_normalize=False, replicated=False):
    """Send batch of feature vectors from CSV (one-per-line, verbose)."""
    samples, lines = load_samples(csv_path, do_normalize)
    if not samples:
        return

    count = 0
    for features in samples:
        count += 1
        print("\n--- Sample %d ---" % count)
        send_features(features, replicated=replicated)
        if delay > 0 and count < len(samples):
            time.sleep(delay)

    print("\nSent %d samples (delay=%.2fs)" % (count, delay))

    if lines and "true_label" in lines[0]:
        print("\nFor accuracy comparison on nf5:")
        print("  scp %s nf5:~/zil_testing_folder/" % csv_path)
        print("  python lab10reg.py ids_batch_packets %d %s" % (
            count, os.path.basename(csv_path)))


def _drain_responses(sock, bucket, poll_seconds):
    """Non-blocking read of all pending UDP responses on `sock` within
    `poll_seconds`. Appends (timestamp, payload) tuples to `bucket`.
    Returns when select() reports no more data ready."""
    deadline = time.time() + poll_seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            remaining = 0
        r, _, _ = select.select([sock], [], [], remaining)
        if not r:
            return
        try:
            data, addr = sock.recvfrom(2048)
            bucket.append((time.time(), data, addr))
        except socket.error:
            return


def _parse_response_bf16_pair(data):
    """Extract (logit0_bf16, logit1_bf16) from a response payload, auto-
    detecting the orchestrator layout (PCI vs ARM).

    The UDP payload after Eth/IP/UDP starts with the 6-byte node1 padding,
    then BRAM[7..N] of the FPGA response. Two layouts exist:

    - PCI orchestrator (lab10reg.py:_form_and_drain_response) packs both
      logits into a single BRAM word and zeros the next:
        BRAM[7] = {logit0_bf16, logit1_bf16, pred, 0x0000}
        BRAM[8] = 0
      So in payload offsets:
        payload[6..7]  = logit0_bf16
        payload[8..9]  = logit1_bf16
        payload[14..15] = 0x0000  (BRAM[8] zeroed)

    - ARM orchestrator (ids_orchestrator.hex) replicates each logit 4x
      across separate BRAM words (matching its DMA-from-DMEM[126..127]):
        BRAM[7] = {logit0_bf16 x 4 lanes}
        BRAM[8] = {logit1_bf16 x 4 lanes}
      So in payload offsets:
        payload[6..7]   = logit0_bf16
        payload[8..9]   = logit0_bf16  (replicated)
        payload[14..15] = logit1_bf16

    Auto-detect: if payload[14..15] == 0x0000, we're in PCI mode (BRAM[8]
    zeroed); use payload[8..9] for logit1. Otherwise we're in ARM mode;
    use payload[14..15].

    Returns (logit0_bf16, logit1_bf16) or None if the payload is too
    short. Does not compute argmax -- caller can do that.
    """
    if len(data) < 22:
        return None
    def _b(x):
        # Python 2: str-indexed byte is a 1-char str; Python 3: int.
        if isinstance(x, str):
            return ord(x)
        return x
    try:
        b6  = _b(data[6])
        b7  = _b(data[7])
        b8  = _b(data[8])
        b9  = _b(data[9])
        b14 = _b(data[14])
        b15 = _b(data[15])
    except (IndexError, TypeError):
        return None
    logit0     = (b6  << 8) | b7
    pci_logit1 = (b8  << 8) | b9
    arm_logit1 = (b14 << 8) | b15
    # Auto-detect: PCI zeros BRAM[8]; ARM replicates logit1 there.
    if arm_logit1 == 0:
        logit1 = pci_logit1
    else:
        logit1 = arm_logit1
    return (logit0, logit1)


def send_throughput(samples, count, delay, replicated, listen,
                    grace_seconds=3.0, inline_drain=True):
    """Send `count` packets (cycling through `samples` if needed) and
    optionally listen for responses. Reports send and receive throughput.

    Args:
        samples: list of feature vectors (each length N_FEATURES)
        count:   total packets to send
        delay:   seconds between sends (0 = back-to-back as fast as Python)
        replicated: packet layout flag (forwarded to pack_features_*)
        listen:  if True, bind local port and count incoming responses
        grace_seconds: after all sends, how long to wait for stragglers
        inline_drain: if True, poll for responses between each send
    """
    if not samples:
        print("ERROR: no samples to send")
        return

    # Open socket. If listening, bind to LISTEN_PORT so the sender's source
    # port is stable and the FPGA response (which swaps UDP src/dst) comes
    # back to the same socket.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if listen:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except socket.error:
            pass
        try:
            sock.bind(('', LISTEN_PORT))
        except socket.error:
            e = sys.exc_info()[1]
            print("ERROR: cannot bind UDP port %d (%s); is another sender running?" % (
                LISTEN_PORT, e))
            sock.close()
            return

    responses = []
    sent = 0

    print("Sending %d packets (delay=%.3fs, replicated=%s, listen=%s)..." % (
        count, delay, replicated, listen))

    t_send_start = time.time()
    for i in range(count):
        features = samples[i % len(samples)]
        if replicated:
            data = pack_features_replicated(features)
        else:
            data = pack_features(features)
        payload = PADDING + data
        try:
            sock.sendto(payload, (FPGA_IP, FPGA_PORT))
            sent += 1
        except socket.error:
            e = sys.exc_info()[1]
            print("WARN: sendto failed on packet %d: %s" % (i, e))
            # Brief pause and retry path omitted -- just skip and continue

        if listen and inline_drain:
            _drain_responses(sock, responses, 0.0)

        if delay > 0 and i < count - 1:
            time.sleep(delay)

    t_send_end = time.time()

    # Grace period for late responses
    if listen:
        _drain_responses(sock, responses, grace_seconds)
    t_recv_end = time.time()

    sock.close()

    # -------- Report --------
    send_window = max(t_send_end - t_send_start, 1e-6)
    send_pps = sent / send_window

    print("")
    print("=== send_ids throughput report ===")
    print("  Sent:            %d packets" % sent)
    print("  Send window:     %.3f s" % send_window)
    print("  Send rate:       %.2f pps  (Python sendto pace)" % send_pps)

    if listen:
        recv = len(responses)
        if recv == 0:
            print("  Received:        0 responses  (waited %.1fs grace after last send)" % (
                t_recv_end - t_send_end))
            print("")
            print("  If ARM orchestrator is running: check cpu_status on nf5,")
            print("  confirm FIFO_MODE=0, and that bitfile has Fix B RTL loaded.")
        else:
            t_first = responses[0][0]
            t_last  = responses[-1][0]
            recv_window = max(t_last - t_send_start, 1e-6)
            recv_pps = recv / recv_window
            first_logits = _parse_response_bf16_pair(responses[0][1])
            if sent > 0:
                pct = 100.0 * recv / sent
            else:
                pct = 0.0
            print("  Received:        %d responses  (%.1f%% of sent)" % (recv, pct))
            print("  First response:  +%.3f s after first send" % (t_first - t_send_start))
            print("  Last response:   +%.3f s after first send" % (t_last  - t_send_start))
            print("  End-to-end pps:  %.2f  (responses received / total window)" % recv_pps)
            if first_logits is not None:
                lo0, lo1 = first_logits
                f0 = bf16_to_float(lo0)
                f1 = bf16_to_float(lo1)
                if f0 > f1:
                    label = "NORMAL"
                else:
                    label = "ATTACK"
                print("  First logits:    BF16 0x%04X / 0x%04X  (~%.2f, ~%.2f)  -> %s" % (
                    lo0, lo1, f0, f1, label))


def main():
    args = sys.argv[1:]

    # Flag extraction (order-independent, can coexist with --csv and direct features)
    replicated = False
    listen = False
    if '--replicated' in args:
        replicated = True
        args = [a for a in args if a != '--replicated']
    if '--listen' in args:
        listen = True
        args = [a for a in args if a != '--listen']

    # --count N: throughput-style send of N packets (cycles CSV or uses defaults)
    count = None
    for i, a in enumerate(args):
        if a == '--count' and i + 1 < len(args):
            count = int(args[i + 1])
            break
    if count is not None:
        args = [a for idx, a in enumerate(args)
                if a != '--count' and (idx == 0 or args[idx - 1] != '--count')]

    # --delay D: seconds between sends
    delay = None
    for i, a in enumerate(args):
        if a == '--delay' and i + 1 < len(args):
            delay = float(args[i + 1])
            break
    if delay is not None:
        args = [a for idx, a in enumerate(args)
                if a != '--delay' and (idx == 0 or args[idx - 1] != '--delay')]

    if not args:
        if count is not None or listen:
            # Throughput-style with default sample
            samples = [DEFAULT_FEATURES]
            if count is not None:
                eff_count = count
            else:
                eff_count = 1
            if delay is not None:
                eff_delay = delay
            else:
                eff_delay = 0.0
            send_throughput(samples, eff_count, eff_delay, replicated, listen)
        else:
            # Legacy default: single send of attack sample
            if replicated:
                layout_note = " (replicated)"
            else:
                layout_note = ""
            print("Sending default attack sample (normalized features)%s" % layout_note)
            send_features(DEFAULT_FEATURES, replicated=replicated)
        return

    if args[0] == '--csv':
        if len(args) < 2:
            print("Usage: send_ids.py --csv <file.csv> [--count N] [--delay S] [--norm] [--replicated] [--listen]")
            return
        csv_path = args[1]
        do_norm = False
        for i in range(2, len(args)):
            if args[i] == '--norm':
                do_norm = True
        samples, lines = load_samples(csv_path, do_norm)
        if not samples:
            return
        if count is not None or listen:
            # Throughput mode: cycle through samples, exact count
            if count is not None:
                eff_count = count
            else:
                eff_count = len(samples)
            if delay is not None:
                eff_delay = delay
            else:
                eff_delay = 0.0
            send_throughput(samples, eff_count, eff_delay, replicated, listen)
        else:
            # Legacy verbose mode: one send per CSV row
            if delay is not None:
                eff_delay = delay
            else:
                eff_delay = 0.5
            send_batch(csv_path, eff_delay, do_norm, replicated=replicated)
        return

    if args[0] in ('--help', '-h', 'help'):
        print("send_ids.py -- Send IDS feature vectors to NetFPGA")
        print("")
        print("Usage:")
        print("  send_ids.py                                Send default attack sample")
        print("  send_ids.py <f0> <f1> ... <f10>             Send 11 normalized floats")
        print("  send_ids.py --csv <file> [--delay 0.5]      Batch from CSV (0.5s default, verbose)")
        print("  send_ids.py --csv <file> --norm             Batch + auto-normalize")
        print("  send_ids.py [...] --replicated              11 replicated words at BRAM[7..17]")
        print("                                              (for ARM orchestrator DMA path)")
        print("")
        print("Throughput measurement:")
        print("  send_ids.py --csv <file> --count N [--delay 0]  Send exactly N packets,")
        print("                                                  cycling through CSV rows")
        print("  send_ids.py [...] --listen                  Bind local UDP port %d," % LISTEN_PORT)
        print("                                              count responses from FPGA,")
        print("                                              report end-to-end pps")
        print("")
        print("Example (ARM orchestration throughput):")
        print("  python send_ids.py --csv test_features.csv --count 100 --delay 0 --replicated --listen")
        print("")
        print("Layouts:")
        print("  default    -- 3 packed BF16-per-word at BRAM[7..9]")
        print("                (consumed by lab10reg.py ids_batch_packets)")
        print("  replicated -- 11 one-feature-per-word at BRAM[7..17], 4x replicated")
        print("                (consumed by ids_orchestrator.hex direct DMA)")
        return

    # Direct feature values from command line
    if len(args) == N_FEATURES:
        features = [float(a) for a in args]
        if count is not None or listen:
            if count is not None:
                eff_count = count
            else:
                eff_count = 1
            if delay is not None:
                eff_delay = delay
            else:
                eff_delay = 0.0
            send_throughput([features], eff_count, eff_delay, replicated, listen)
        else:
            send_features(features, replicated=replicated)
    else:
        print("ERROR: expected %d feature values, got %d" % (N_FEATURES, len(args)))
        print("Use --help for usage information")


if __name__ == '__main__':
    main()
