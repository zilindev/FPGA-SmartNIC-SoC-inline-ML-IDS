#!/usr/bin/env python3
"""
validate_ids_csv.py -- BF16 forward pass on a normalized features CSV.

Replicates exactly what the FPGA kernel would compute (BF16 truncation per
FMA), then compares against the `true_label` column. Use this to predict
FPGA-side demo accuracy before deploying.

Usage:
  py -3 scripts/validate_ids_csv.py [--csv scripts/test_features.csv] [--tag trained]
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'toolchains', 'gpu'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ids_v2 import evaluate_bf16_full, fmt_cm


def load_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    header = [h.strip() for h in lines[0].strip().split(',')]
    if header[-1] != 'true_label':
        raise ValueError("CSV missing trailing 'true_label' column")
    feat_names = header[:-1]
    rows = []
    labels = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != len(header):
            continue
        rows.append([float(p) for p in parts[:-1]])
        labels.append(int(parts[-1]))
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, feat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='scripts/test_features.csv')
    ap.add_argument('--tag', type=str, default='trained')
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    weights_path = os.path.join(project_dir, 'programs', 'gpu',
                                'data_ids_%s_weights.json' % args.tag)
    with open(weights_path, 'r') as f:
        meta = json.load(f)
    weights = meta['weights']

    X, y, feat_names = load_csv(args.csv)
    expected = meta['selected_features']
    if feat_names != expected:
        print("WARN: feature columns differ from training metadata.")
        print("  csv:      %s" % feat_names)
        print("  expected: %s" % expected)

    print("Running BF16 forward (HW-faithful FMA semantics) on %d samples..."
          % len(X))
    acc, cm, preds, logits = evaluate_bf16_full(X, y, weights)
    print("\nBF16 accuracy: %.4f (%d/%d)" % (acc, int(round(acc * len(X))), len(X)))
    print("\nConfusion matrix:")
    print(fmt_cm(cm))
    print("\nPer-sample:")
    print("  idx  true  pred  logit0     logit1     ok")
    for i in range(len(X)):
        ok = " " if preds[i] == y[i] else "X"
        print("  %3d  %4d  %4d  %9.4f  %9.4f  %s"
              % (i, y[i], preds[i], logits[i, 0], logits[i, 1], ok))


if __name__ == '__main__':
    main()
