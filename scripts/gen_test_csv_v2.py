#!/usr/bin/env python3
"""
gen_test_csv_v2.py -- Generate normalized test CSV for the v2 IDS pipeline.

Mirrors the train_ids_v2.py preprocessing exactly:
  1. Build categorical -> int maps from KDDTrain+ (must match training).
  2. Load KDDTest+, including label-encoded categorical features if they were
     used in training.
  3. Apply log1p transform on the same heavy-tailed columns the trainer used.
  4. Select columns by `selected_indices` from the trained weights JSON.
  5. Min-max with the trained norm_params.
  6. Pick a balanced subset of N samples (N/2 normal + N/2 attack), shuffle.
  7. Write CSV with the selected feature names + true_label column.

Usage:
  py -3 scripts/gen_test_csv_v2.py [-n 20] [-o test_features.csv]
                                   [--tag trained]
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'toolchains', 'gpu'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ids_v2 import (
    NSL_COLUMNS, CATEGORICAL_RAW, LABEL_COL, NORMAL_LABELS, LOG_FEATURES,
    build_categorical_maps, load_nslkdd, apply_log_transform,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n-samples', type=int, default=20)
    ap.add_argument('-o', '--output', type=str, default=None)
    ap.add_argument('--tag', type=str, default='trained',
                    help="Read from data_ids_<tag>_weights.json")
    ap.add_argument('--shuffle-seed', type=int, default=42)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if args.output is None:
        args.output = os.path.join(script_dir, 'test_features.csv')

    weights_path = os.path.join(project_dir, 'programs', 'gpu',
                                'data_ids_%s_weights.json' % args.tag)
    with open(weights_path, 'r') as f:
        meta = json.load(f)

    selected_features = meta['selected_features']
    selected_indices = meta['selected_indices']
    mins = np.array(meta['norm_params']['mins'], dtype=np.float32)
    maxs = np.array(meta['norm_params']['maxs'], dtype=np.float32)
    rng = (maxs - mins)
    rng[rng == 0] = 1.0

    pre = meta.get('preprocessing', {})
    use_log = bool(pre.get('log_features'))
    log_set = set(pre.get('log_features') or LOG_FEATURES)
    include_cat = bool(pre.get('include_categorical'))
    saved_cat_maps = pre.get('categorical_maps') or {}

    # Categorical maps must match training. Prefer the maps saved with the
    # trained model; fall back to rebuilding from train file if missing.
    train_path = os.path.join(project_dir, 'datasets', 'KDDTrain+.txt')
    test_path = os.path.join(project_dir, 'datasets', 'KDDTest+.txt')
    if include_cat and saved_cat_maps:
        cat_maps = {c: dict(saved_cat_maps[c]) for c in CATEGORICAL_RAW
                    if c in saved_cat_maps}
        # Stored map values are ints; convert
        for c in cat_maps:
            cat_maps[c] = {k: int(v) for k, v in cat_maps[c].items()}
    elif include_cat:
        cat_maps = build_categorical_maps(train_path)
    else:
        cat_maps = {c: {} for c in CATEGORICAL_RAW}

    # Build test feature matrix using the SAME pipeline as training
    X_test, y_test, col_names = load_nslkdd(test_path, cat_maps, include_cat)
    if use_log:
        X_test = apply_log_transform(X_test, col_names, log_set)

    # Sanity: column names from loader should match saved selected_features
    # at the saved indices.
    for i, sidx in enumerate(selected_indices):
        if col_names[sidx] != selected_features[i]:
            print("WARN: column mismatch at slot %d: meta=%r loader=%r"
                  % (i, selected_features[i], col_names[sidx]))

    X_sel = X_test[:, selected_indices]
    X_norm = np.clip((X_sel - mins) / rng, 0.0, 1.0)

    # Balanced subset
    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    n_normal = args.n_samples // 2
    n_attack = args.n_samples - n_normal

    sub = np.concatenate([normal_idx[:n_normal], attack_idx[:n_attack]])
    rs = np.random.RandomState(args.shuffle_seed)
    rs.shuffle(sub)

    with open(args.output, 'w') as f:
        f.write(','.join(selected_features) + ',true_label\n')
        for idx in sub:
            row = X_norm[idx]
            f.write(','.join('%.6f' % v for v in row))
            f.write(',%d\n' % int(y_test[idx]))

    print("Wrote %d samples -> %s" % (len(sub), args.output))
    print("  Normal: %d  Attack: %d" % (
        int(np.sum(y_test[sub] == 0)),
        int(np.sum(y_test[sub] == 1))))
    print("  Features: %s" % ', '.join(selected_features))


if __name__ == '__main__':
    main()
