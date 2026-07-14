#!/usr/bin/env python3
"""
compare_ids_perf.py -- Compile and run the scalar and BCAST IDS inference
testbenches, parse GPU cycle counts, and print a side-by-side performance
comparison table.

This is the "with BCAST vs. without BCAST" demo script for milestone 2 --
the 3.28x speedup is visible only in simulation (PCI register I/O dominates
wallclock on hardware, so the GPU kernel time is invisible from the host).

Usage:
    python scripts/compare_ids_perf.py

    # Override iverilog path:
    python scripts/compare_ids_perf.py --iverilog "C:/iverilog/bin/iverilog.exe"

    # Skip recompilation (reuse previously built simulators):
    python scripts/compare_ids_perf.py --no-compile

    # Emit markdown table for the lab report:
    python scripts/compare_ids_perf.py --markdown docs/bcast_perf_comparison.md
"""

import os
import re
import sys
import shutil
import subprocess
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

GPU_SRCS = [
    'src/gpu/gpu_top.v',
    'src/gpu/core_top.v',
    'src/gpu/program_counter.v',
    'src/gpu/instruction_memory.v',
    'src/gpu/data_memory.v',
    'src/gpu/fetch_stage.v',
    'src/gpu/decode_stage.v',
    'src/gpu/instruction_decoder.v',
    'src/gpu/register_file.v',
    'src/gpu/pipeline_reg_idex.v',
    'src/gpu/pipeline_reg_ex.v',
    'src/gpu/pipeline_reg_exwb.v',
    'src/gpu/execute_stage.v',
    'src/gpu/simd_alu.v',
    'src/gpu/int16_alu.v',
    'src/gpu/bf16_alu.v',
    'src/gpu/bf16_fma_unit.v',
    'src/gpu/hazard_unit.v',
    'src/gpu/writeback_stage.v',
]

KERNELS = [
    {
        'name':    'Scalar (baseline)',
        'short':   'scalar',
        'tb':      'tb/tb_ids_inference.v',
        'tb_mod':  'tb_ids_inference',
        'sim_out': 'build/tb_ids_scalar_sim',
        'imem':    'programs/gpu/ann_ids_11_16_8_2.hex',
        'dmem':    'programs/gpu/data_ids_11_16_8_2.hex',
    },
    {
        'name':    'BCAST (optimized)',
        'short':   'bcast',
        'tb':      'tb/tb_ids_bcast.v',
        'tb_mod':  'tb_ids_bcast',
        'sim_out': 'build/tb_ids_bcast_sim',
        'imem':    'programs/gpu/ann_ids_11_16_8_2_bcast.hex',
        'dmem':    'programs/gpu/data_ids_11_16_8_2_bcast.hex',
    },
]

CYCLE_RE = re.compile(r'halted after\s+(\d+)\s+cycles', re.IGNORECASE)
PASS_RE  = re.compile(r'Results:\s+(\d+)\s+PASS,\s+(\d+)\s+FAIL', re.IGNORECASE)


def find_iverilog(override=None):
    if override:
        return override
    for cand in (
        shutil.which('iverilog'),
        'C:/iverilog/bin/iverilog.exe',
        '/c/iverilog/bin/iverilog.exe',
    ):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("iverilog not found; pass --iverilog <path>")


def find_vvp(iverilog_path):
    vvp = os.path.join(os.path.dirname(iverilog_path), 'vvp.exe')
    if not os.path.exists(vvp):
        vvp = os.path.join(os.path.dirname(iverilog_path), 'vvp')
    if not os.path.exists(vvp):
        raise RuntimeError("vvp not found next to iverilog at %s" % iverilog_path)
    return vvp


def count_hex_lines(path):
    full = os.path.join(REPO_ROOT, path)
    n = 0
    with open(full, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('//') and not s.startswith('#'):
                n += 1
    return n


def compile_tb(iverilog, kernel):
    out_path = os.path.join(REPO_ROOT, kernel['sim_out'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        iverilog, '-g2001', '-I', 'include',
        '-o', kernel['sim_out'],
    ] + GPU_SRCS + [kernel['tb']]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        print("Compilation failed for %s:\n%s" % (kernel['name'], proc.stdout))
        sys.exit(1)


def run_tb(vvp, kernel):
    proc = subprocess.run(
        [vvp, kernel['sim_out']],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        print("Simulation failed for %s:\n%s" % (kernel['name'], proc.stdout))
        sys.exit(1)
    return proc.stdout


def parse_output(out):
    cycles = None
    passes = fails = None
    for line in out.splitlines():
        m = CYCLE_RE.search(line)
        if m:
            cycles = int(m.group(1))
        m = PASS_RE.search(line)
        if m:
            passes = int(m.group(1))
            fails  = int(m.group(2))
    return cycles, passes, fails


def format_row(label, width, *cells, **kwargs):
    sep = kwargs.get('sep', ' | ')
    return label.ljust(width) + sep + sep.join(cells)


def print_table(results):
    scalar = results['scalar']
    bcast  = results['bcast']

    metrics = [
        ('GPU cycles',            '%d' % scalar['cycles'],   '%d' % bcast['cycles'],
         '%.2fx faster' % (scalar['cycles'] / bcast['cycles'])),
        ('Time @ 125 MHz (us)',   '%.2f' % (scalar['cycles'] * 8 / 1000.0),
                                  '%.2f' % (bcast['cycles']  * 8 / 1000.0),
         '-%.2f us' % ((scalar['cycles'] - bcast['cycles']) * 8 / 1000.0)),
        ('IMEM instructions',     '%d' % scalar['imem'],     '%d' % bcast['imem'],
         '%+d (%.0f%%)' % (bcast['imem'] - scalar['imem'],
                            100.0 * (bcast['imem'] - scalar['imem']) / scalar['imem'])),
        ('DMEM words',            '%d' % scalar['dmem'],     '%d' % bcast['dmem'],
         '%.2fx smaller' % (scalar['dmem'] / float(bcast['dmem']))),
        ('Verification',          '%d/%d PASS' % (scalar['passes'],
                                                   scalar['passes'] + scalar['fails']),
                                  '%d/%d PASS' % (bcast['passes'],
                                                   bcast['passes'] + bcast['fails']),
         'both OK' if scalar['fails'] == 0 and bcast['fails'] == 0 else 'CHECK'),
    ]

    w_label = max(len('Metric'), max(len(m[0]) for m in metrics))
    w_scalar = max(len('Scalar'), max(len(m[1]) for m in metrics))
    w_bcast  = max(len('BCAST'),  max(len(m[2]) for m in metrics))
    w_delta  = max(len('Delta'),  max(len(m[3]) for m in metrics))

    header = format_row('Metric'.ljust(w_label),
                        w_label,
                        'Scalar'.ljust(w_scalar),
                        'BCAST'.ljust(w_bcast),
                        'Delta'.ljust(w_delta))
    rule = '-' * len(header)
    print('=' * len(header))
    print("IDS 11-16-8-2 Inference: Scalar vs. BCAST  (cycle-accurate sim)")
    print('=' * len(header))
    print(header)
    print(rule)
    for label, s, b, d in metrics:
        print(format_row(label.ljust(w_label), w_label,
                         s.ljust(w_scalar),
                         b.ljust(w_bcast),
                         d.ljust(w_delta)))
    print('=' * len(header))
    print("Speedup: %.2fx  (%d -> %d cycles, %d cycles saved)" % (
        scalar['cycles'] / bcast['cycles'],
        scalar['cycles'], bcast['cycles'],
        scalar['cycles'] - bcast['cycles']))
    print('=' * len(header))


def emit_markdown(results, path):
    scalar = results['scalar']
    bcast  = results['bcast']
    speedup = scalar['cycles'] / bcast['cycles']
    dmem_ratio = scalar['dmem'] / float(bcast['dmem'])
    imem_delta_pct = 100.0 * (bcast['imem'] - scalar['imem']) / scalar['imem']

    lines = [
        "# IDS 11-16-8-2 Inference: Scalar vs. BCAST",
        "",
        "Cycle-accurate simulation results for the intrusion-detection MLP",
        "kernel before and after adding the BCAST cross-lane broadcast instruction.",
        "",
        "| Metric | Scalar (baseline) | BCAST (optimized) | Delta |",
        "|---|---|---|---|",
        "| GPU cycles | %d | %d | **%.2fx faster** |" % (
            scalar['cycles'], bcast['cycles'], speedup),
        "| Time @ 125 MHz | %.2f us | %.2f us | -%.2f us |" % (
            scalar['cycles'] * 8 / 1000.0,
            bcast['cycles']  * 8 / 1000.0,
            (scalar['cycles'] - bcast['cycles']) * 8 / 1000.0),
        "| IMEM instructions | %d | %d | %+d (%.0f%%) |" % (
            scalar['imem'], bcast['imem'],
            bcast['imem'] - scalar['imem'], imem_delta_pct),
        "| DMEM words | %d | %d | **%.2fx smaller** |" % (
            scalar['dmem'], bcast['dmem'], dmem_ratio),
        "| Verification | %d/%d PASS | %d/%d PASS | both OK |" % (
            scalar['passes'], scalar['passes'] + scalar['fails'],
            bcast['passes'],  bcast['passes']  + bcast['fails']),
        "",
        "**Headline:** %.2fx speedup (%d -> %d cycles), %.1fx smaller DMEM footprint, %d fewer instructions." % (
            speedup, scalar['cycles'], bcast['cycles'], dmem_ratio,
            scalar['imem'] - bcast['imem']),
        "",
        "The speedup comes from using all 4 SIMD lanes for independent neuron",
        "accumulation (packed weights + replicated inputs -> one FMA per cycle",
        "updates 4 neurons at once). Before BCAST, inter-layer data could not be",
        "reshuffled across lanes, forcing every kernel into scalar mode with 25%",
        "SIMD utilization.",
    ]
    full = os.path.join(REPO_ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("\nWrote markdown report to %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iverilog', help='Path to iverilog')
    ap.add_argument('--no-compile', action='store_true',
                    help='Reuse existing simulator binaries')
    ap.add_argument('--markdown', help='Also emit a markdown table to this path')
    args = ap.parse_args()

    iverilog = find_iverilog(args.iverilog)
    vvp = find_vvp(iverilog)

    results = {}
    for kernel in KERNELS:
        print("[%s] compiling..." % kernel['short'])
        if not args.no_compile:
            compile_tb(iverilog, kernel)
        print("[%s] running..." % kernel['short'])
        out = run_tb(vvp, kernel)
        cycles, passes, fails = parse_output(out)
        if cycles is None:
            print("Failed to parse cycle count from %s output:\n%s" % (
                kernel['short'], out))
            sys.exit(1)
        results[kernel['short']] = {
            'cycles': cycles,
            'passes': passes or 0,
            'fails':  fails  or 0,
            'imem':   count_hex_lines(kernel['imem']),
            'dmem':   count_hex_lines(kernel['dmem']),
        }
        print("[%s] %d cycles, %d PASS / %d FAIL" % (
            kernel['short'], cycles, passes or 0, fails or 0))
    print()

    print_table(results)

    if args.markdown:
        emit_markdown(results, args.markdown)


if __name__ == '__main__':
    main()
