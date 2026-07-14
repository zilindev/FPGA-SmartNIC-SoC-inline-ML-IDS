#!/usr/bin/env python3
"""
gen_test_csv.py -- Generate test CSV file with normalized IDS features.

Extracts N samples from NSL-KDD test set using the trained model's
feature selection and normalization. For use with send_ids.py --csv
and lab10reg.py ids_batch.

Usage:
    python gen_test_csv.py [-n 20] [-o test_features.csv]
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'toolchains', 'gpu'))
from bf16_utils import float_to_bf16, bf16_to_float

# NSL-KDD columns (same as train_ids.py)
NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
CATEGORICAL = {'protocol_type', 'service', 'flag', 'label'}
NORMAL_LABELS = {'normal'}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    n_samples = 20
    output_path = os.path.join(script_dir, 'test_features.csv')

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '-n' and i + 1 < len(args):
            n_samples = int(args[i + 1])
        if a == '-o' and i + 1 < len(args):
            output_path = args[i + 1]

    # Load trained model info
    weights_path = os.path.join(project_dir, 'programs', 'gpu',
                                'data_ids_trained_weights.json')
    with open(weights_path, 'r') as f:
        model_info = json.load(f)

    selected_features = model_info['selected_features']
    selected_indices = model_info['selected_indices']
    norm_params = model_info['norm_params']
    mins = np.array(norm_params['mins'])
    maxs = np.array(norm_params['maxs'])
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    # Load test data
    test_path = os.path.join(project_dir, 'datasets', 'KDDTest+.txt')
    numeric_cols = [c for c in NSL_COLUMNS[:41] if c not in CATEGORICAL]

    data = []
    labels = []
    with open(test_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 42:
                continue
            row = {}
            for i, col in enumerate(NSL_COLUMNS[:len(parts)]):
                row[col] = parts[i]
            label_str = row['label'].strip().rstrip('.')
            labels.append(0 if label_str in NORMAL_LABELS else 1)
            numeric_row = []
            for col in numeric_cols:
                try:
                    numeric_row.append(float(row[col]))
                except (ValueError, KeyError):
                    numeric_row.append(0.0)
            data.append(numeric_row)

    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Select and normalize
    X_sel = X[:, selected_indices]
    X_norm = (X_sel - mins) / ranges
    X_norm = np.clip(X_norm, 0.0, 1.0)

    # Sample: mix of normal and attack
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    n_normal = n_samples // 2
    n_attack = n_samples - n_normal
    chosen = np.concatenate([
        normal_idx[:n_normal],
        attack_idx[:n_attack]
    ])
    np.random.seed(42)
    np.random.shuffle(chosen)

    # Write CSV
    with open(output_path, 'w') as f:
        # Header
        f.write(','.join(selected_features) + ',true_label\n')
        for idx in chosen:
            vals = X_norm[idx]
            f.write(','.join(['%.6f' % v for v in vals]))
            f.write(',%d\n' % y[idx])

    print("Generated %d samples -> %s" % (len(chosen), output_path))
    print("  Normal: %d, Attack: %d" % (
        sum(1 for i in chosen if y[i] == 0),
        sum(1 for i in chosen if y[i] == 1)))
    print("  Features: %s" % ', '.join(selected_features))


if __name__ == '__main__':
    main()
