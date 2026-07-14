#!/usr/bin/env python3
"""
train_ids.py -- Train IDS MLP (11->16->8->2) on NSL-KDD, export BF16 weights.

Downloads NSL-KDD if not present, selects top 11 numerical features by
Random Forest importance, trains a 3-layer ReLU MLP with float32, quantizes
weights to BF16, and exports for the GPU kernel.

Usage:
    python train_ids.py                    # Train + export
    python train_ids.py --eval-only        # Load existing weights, evaluate
    python train_ids.py --features 11      # Override feature count

Output:
    datasets/KDDTrain+.txt                 # Downloaded dataset
    datasets/KDDTest+.txt
    programs/gpu/data_ids_trained.hex      # DMEM hex for GPU
    programs/gpu/data_ids_trained_weights.json
    programs/gpu/data_ids_trained_expected.txt
"""

import os
import sys
import json
import struct
import urllib.request
import numpy as np

# Add toolchain path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'toolchains', 'gpu'))
from bf16_utils import float_to_bf16, bf16_to_float, pack_bf16_vector, format_hex64

# =====================================================================
# NSL-KDD column definitions
# =====================================================================
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

# Categorical columns (cannot use directly as numeric features)
CATEGORICAL = {'protocol_type', 'service', 'flag', 'label'}

# Attack type -> binary label mapping
NORMAL_LABELS = {'normal'}


# =====================================================================
# Dataset download
# =====================================================================
def download_nslkdd(data_dir):
    """Download NSL-KDD train and test files if not present."""
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = {
        'KDDTrain+.txt': base_url + 'KDDTrain%2B.txt',
        'KDDTest+.txt': base_url + 'KDDTest%2B.txt',
    }

    os.makedirs(data_dir, exist_ok=True)
    for fname, url in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"  {fname} already exists ({os.path.getsize(path)} bytes)")
            continue
        print(f"  Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Saved {fname} ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")
            print(f"  Trying alternate URL...")
            # Alternate: University of New Brunswick mirror
            alt_url = f"https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/{fname}"
            try:
                urllib.request.urlretrieve(alt_url, path)
                print(f"  Saved {fname} from alternate ({os.path.getsize(path)} bytes)")
            except Exception as e2:
                print(f"  ERROR: Could not download {fname}: {e2}")
                return False
    return True


# =====================================================================
# Data loading and preprocessing
# =====================================================================
def load_nslkdd(filepath):
    """Load NSL-KDD CSV file into numpy arrays."""
    data = []
    labels = []
    with open(filepath, 'r') as f:
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

            # Binary label: normal=0, attack=1
            label_str = row['label'].strip().rstrip('.')
            label = 0 if label_str in NORMAL_LABELS else 1
            labels.append(label)

            # Extract numeric features only
            numeric_row = []
            for col in NSL_COLUMNS[:41]:
                if col in CATEGORICAL:
                    continue
                try:
                    numeric_row.append(float(row[col]))
                except (ValueError, KeyError):
                    numeric_row.append(0.0)
            data.append(numeric_row)

    # Build numeric column name list
    numeric_cols = [c for c in NSL_COLUMNS[:41] if c not in CATEGORICAL]

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32), numeric_cols


def select_features(X_train, y_train, col_names, n_features=11):
    """Select top N features by Random Forest importance."""
    from sklearn.ensemble import RandomForestClassifier

    print(f"\nSelecting top {n_features} features by RF importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    #print("  Feature importance ranking:")
    for i in range(min(20, len(col_names))):
        idx = indices[i]
        #marker = " <--" if i < n_features else ""
        #print(f"    {i+1:2d}. {col_names[idx]:30s} {importances[idx]:.4f}{marker}")

    selected_indices = indices[:n_features]
    selected_names = [col_names[i] for i in selected_indices]
    print(f"\n  Selected: {selected_names}")

    return selected_indices, selected_names


def normalize_features(X_train, X_test, selected_indices):
    """Min-max normalize selected features to [0, 1] range."""
    X_tr = X_train[:, selected_indices].copy()
    X_te = X_test[:, selected_indices].copy()

    mins = X_tr.min(axis=0)
    maxs = X_tr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # Avoid division by zero

    X_tr = (X_tr - mins) / ranges
    X_te = (X_te - mins) / ranges

    # Clip test set to [0, 1] (may have out-of-range values)
    X_te = np.clip(X_te, 0.0, 1.0)

    norm_params = {'mins': mins.tolist(), 'maxs': maxs.tolist()}
    return X_tr, X_te, norm_params


# =====================================================================
# MLP training (manual numpy implementation, no PyTorch dependency)
# =====================================================================
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    n = len(labels)
    clipped = np.clip(probs[np.arange(n), labels], 1e-7, 1.0)
    return -np.mean(np.log(clipped))


def train_mlp(X_train, y_train, X_test, y_test,
              hidden1=16, hidden2=8, n_classes=2,
              lr=0.01, epochs=50, batch_size=256):
    """Train 3-layer MLP with SGD + momentum."""
    n_features = X_train.shape[1]
    np.random.seed(42)

    # Xavier initialization
    W1 = np.random.randn(n_features, hidden1).astype(np.float32) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
    b2 = np.zeros(hidden2, dtype=np.float32)
    W3 = np.random.randn(hidden2, n_classes).astype(np.float32) * np.sqrt(2.0 / hidden2)
    b3 = np.zeros(n_classes, dtype=np.float32)

    # Momentum
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
    vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)
    momentum = 0.9

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            bs = end - start

            # Forward
            z1 = X_batch @ W1 + b1
            h1 = relu(z1)
            z2 = h1 @ W2 + b2
            h2 = relu(z2)
            z3 = h2 @ W3 + b3
            probs = softmax(z3)

            epoch_loss += cross_entropy_loss(probs, y_batch) * bs

            # Backward
            dz3 = probs.copy()
            dz3[np.arange(bs), y_batch] -= 1
            dz3 /= bs

            dW3 = h2.T @ dz3
            db3 = dz3.sum(axis=0)
            dh2 = dz3 @ W3.T
            dz2 = dh2 * relu_grad(z2)

            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)
            dh1 = dz2 @ W2.T
            dz1 = dh1 * relu_grad(z1)

            dW1 = X_batch.T @ dz1
            db1 = dz1.sum(axis=0)

            # SGD + momentum
            for param, grad, vel in [
                (W1, dW1, vW1), (b1, db1, vb1),
                (W2, dW2, vW2), (b2, db2, vb2),
                (W3, dW3, vW3), (b3, db3, vb3),
            ]:
                vel[:] = momentum * vel - lr * grad
                param += vel

            n_batches += 1

        epoch_loss /= n_samples

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = evaluate_mlp(X_train, y_train, W1, b1, W2, b2, W3, b3)
            test_acc = evaluate_mlp(X_test, y_test, W1, b1, W2, b2, W3, b3)
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    weights = {
        'w1': W1.T.tolist(),  # Transpose: [hidden1, n_features] for kernel layout
        'b1': b1.tolist(),
        'w2': W2.T.tolist(),  # [hidden2, hidden1]
        'b2': b2.tolist(),
        'w3': W3.T.tolist(),  # [n_classes, hidden2]
        'b3': b3.tolist(),
    }

    return weights


def evaluate_mlp(X, y, W1, b1, W2, b2, W3, b3):
    """Forward pass and accuracy computation."""
    h1 = relu(X @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    z3 = h2 @ W3 + b3
    preds = np.argmax(z3, axis=1)
    return np.mean(preds == y)


def forward_single_bf16(x, weights):
    """Forward pass for a single sample using BF16 FMA arithmetic.
    Matches hardware: FMA computes a*b+c in extended precision, then
    truncates to BF16 once (not separate truncation after mul and add)."""
    n_in = len(x)

    # Quantize input
    x_q = [bf16_to_float(float_to_bf16(v)) for v in x]

    # Layer 1: h1 = ReLU(W1 @ x + b1)
    h1 = []
    for n in range(len(weights['b1'])):
        acc = bf16_to_float(float_to_bf16(weights['b1'][n]))
        for i in range(n_in):
            w = bf16_to_float(float_to_bf16(weights['w1'][n][i]))
            xi = bf16_to_float(float_to_bf16(x_q[i]))
            # FMA: single truncation after multiply-add
            acc = bf16_to_float(float_to_bf16(w * xi + acc))
        h1.append(max(0.0, acc))

    # Layer 2: h2 = ReLU(W2 @ h1 + b2)
    h2 = []
    for n in range(len(weights['b2'])):
        acc = bf16_to_float(float_to_bf16(weights['b2'][n]))
        for i in range(len(h1)):
            w = bf16_to_float(float_to_bf16(weights['w2'][n][i]))
            hi = bf16_to_float(float_to_bf16(h1[i]))
            acc = bf16_to_float(float_to_bf16(w * hi + acc))
        h2.append(max(0.0, acc))

    # Layer 3: out = W3 @ h2 + b3 (no activation)
    out = []
    for n in range(len(weights['b3'])):
        acc = bf16_to_float(float_to_bf16(weights['b3'][n]))
        for i in range(len(h2)):
            w = bf16_to_float(float_to_bf16(weights['w3'][n][i]))
            hi = bf16_to_float(float_to_bf16(h2[i]))
            acc = bf16_to_float(float_to_bf16(w * hi + acc))
        out.append(acc)

    return out


def evaluate_bf16(X, y, weights, max_samples=1000):
    """Evaluate accuracy using BF16-quantized forward pass."""
    n = min(len(X), max_samples)
    correct = 0
    for i in range(n):
        out = forward_single_bf16(X[i], weights)
        pred = 0 if out[0] > out[1] else 1
        if pred == y[i]:
            correct += 1
    return correct / n


# =====================================================================
# Export to GPU DMEM format
# =====================================================================
def replicate_bf16(value):
    """Pack a single float as replicated BF16 across all 4 SIMD lanes."""
    return pack_bf16_vector([value, value, value, value])


def build_dmem(inputs, weights):
    """Build 383-word DMEM array matching kernel layout."""
    DMEM_SIZE = 383
    dmem = [0] * DMEM_SIZE

    L1_IN, L1_OUT = 11, 16
    L2_IN, L2_OUT = 16, 8
    L3_IN, L3_OUT = 8, 2

    X_BASE = 0
    W1_BASE = 11
    B1_BASE = 187
    W2_BASE = 219
    B2_BASE = 347
    W3_BASE = 363
    B3_BASE = 379

    # Inputs
    for i in range(L1_IN):
        dmem[X_BASE + i] = replicate_bf16(inputs[i])

    # L1 weights (neuron-major)
    for n in range(L1_OUT):
        for i in range(L1_IN):
            dmem[W1_BASE + n * L1_IN + i] = replicate_bf16(weights['w1'][n][i])
    for n in range(L1_OUT):
        dmem[B1_BASE + n] = replicate_bf16(weights['b1'][n])

    # L2 weights
    for n in range(L2_OUT):
        for i in range(L2_IN):
            dmem[W2_BASE + n * L2_IN + i] = replicate_bf16(weights['w2'][n][i])
    for n in range(L2_OUT):
        dmem[B2_BASE + n] = replicate_bf16(weights['b2'][n])

    # L3 weights
    for n in range(L3_OUT):
        for i in range(L3_IN):
            dmem[W3_BASE + n * L3_IN + i] = replicate_bf16(weights['w3'][n][i])
    for n in range(L3_OUT):
        dmem[B3_BASE + n] = replicate_bf16(weights['b3'][n])

    return dmem


def write_hex(dmem, path):
    """Write DMEM array to $readmemh-compatible hex file."""
    with open(path, 'w') as f:
        for word in dmem:
            f.write(format_hex64(word) + '\n')


# =====================================================================
# Main
# =====================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'datasets')
    gpu_dir = os.path.join(project_dir, 'programs', 'gpu')

    n_features = 11

    # Parse args
    args = sys.argv[1:]
    eval_only = '--eval-only' in args
    for i, a in enumerate(args):
        if a == '--features' and i + 1 < len(args):
            n_features = int(args[i + 1])

    # ----------------------------------------------------------------
    # Download dataset
    # ----------------------------------------------------------------
    print("=" * 60)
    print("NSL-KDD IDS Training Pipeline")
    print("=" * 60)
    print(f"\nModel: {n_features} -> 16 -> 8 -> 2 (binary classification)")
    print(f"\nChecking dataset...")

    if not download_nslkdd(data_dir):
        print("ERROR: Could not download dataset")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print("\nLoading training data...")
    X_train_full, y_train, train_cols = load_nslkdd(
        os.path.join(data_dir, 'KDDTrain+.txt'))
    print(f"  Train: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} numeric features")
    print(f"  Classes: {np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} attack")

    print("\nLoading test data...")
    X_test_full, y_test, _ = load_nslkdd(
        os.path.join(data_dir, 'KDDTest+.txt'))
    print(f"  Test: {X_test_full.shape[0]} samples")
    print(f"  Classes: {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} attack")

    # ----------------------------------------------------------------
    # Feature selection
    # ----------------------------------------------------------------
    selected_idx, selected_names = select_features(
        X_train_full, y_train, train_cols, n_features)

    # ----------------------------------------------------------------
    # Normalize
    # ----------------------------------------------------------------
    print("\nNormalizing features (min-max to [0,1])...")
    X_train, X_test, norm_params = normalize_features(
        X_train_full, X_test_full, selected_idx)
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    if not eval_only:
        print("\nTraining MLP...")
        weights = train_mlp(X_train, y_train, X_test, y_test,
                           hidden1=16, hidden2=8, n_classes=2,
                           lr=0.01, epochs=50, batch_size=256)

        # Save weights
        export_data = {
            'weights': weights,
            'selected_features': selected_names,
            'selected_indices': selected_idx.tolist(),
            'norm_params': norm_params,
            'n_features': n_features,
            'architecture': [n_features, 16, 8, 2],
        }
        weights_path = os.path.join(gpu_dir, 'data_ids_trained_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nSaved weights to {weights_path}")
    else:
        weights_path = os.path.join(gpu_dir, 'data_ids_trained_weights.json')
        with open(weights_path, 'r') as f:
            export_data = json.load(f)
        weights = export_data['weights']
        selected_names = export_data['selected_features']
        print(f"\nLoaded weights from {weights_path}")

    # ----------------------------------------------------------------
    # BF16 quantization accuracy
    # ----------------------------------------------------------------
    print("\nEvaluating BF16-quantized accuracy (first 500 test samples)...")
    bf16_acc = evaluate_bf16(X_test, y_test, weights, max_samples=500)
    print(f"  BF16 test accuracy: {bf16_acc:.4f} ({int(bf16_acc*500)}/500)")

    # ----------------------------------------------------------------
    # Generate test vectors and DMEM hex
    # ----------------------------------------------------------------
    print("\nGenerating GPU DMEM hex files...")

    # Pick a few representative test samples
    test_samples = []
    # One normal, one attack
    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    if len(normal_idx) > 0:
        test_samples.append(('normal', normal_idx[0]))
    if len(attack_idx) > 0:
        test_samples.append(('attack', attack_idx[0]))
    # First sample as default
    test_samples.insert(0, ('default', 0))

    for label, idx in test_samples:
        x = X_test[idx].tolist()
        true_label = y_test[idx]

        # BF16 reference
        out = forward_single_bf16(x, weights)
        pred = 0 if out[0] > out[1] else 1

        #print(f"\n  Sample '{label}' (idx={idx}, true={true_label}, pred={pred}):")
        #print(f"    Input features: {['%.3f' % v for v in x]}")
        #print(f"    Output logits:  {['%.4f' % v for v in out]}")
        #print(f"    Classification: {'normal' if pred == 0 else 'attack'} "
        #      f"({'CORRECT' if pred == true_label else 'WRONG'})")

        if label == 'default':
            # Write the default test file
            dmem = build_dmem(x, weights)
            hex_path = os.path.join(gpu_dir, 'data_ids_trained.hex')
            write_hex(dmem, hex_path)
            #print(f"    Wrote DMEM to {hex_path}")

            # Write expected values
            h1_ref, h2_ref = [], []
            # Recompute layer outputs for expected file (FMA semantics)
            x_q = [bf16_to_float(float_to_bf16(v)) for v in x]
            # Layer 1
            for n in range(16):
                acc = bf16_to_float(float_to_bf16(weights['b1'][n]))
                for i in range(11):
                    w = bf16_to_float(float_to_bf16(weights['w1'][n][i]))
                    xi = bf16_to_float(float_to_bf16(x_q[i]))
                    acc = bf16_to_float(float_to_bf16(w * xi + acc))
                h1_ref.append(max(0.0, acc))
            # Layer 2
            for n in range(8):
                acc = bf16_to_float(float_to_bf16(weights['b2'][n]))
                for i in range(16):
                    w = bf16_to_float(float_to_bf16(weights['w2'][n][i]))
                    hi = bf16_to_float(float_to_bf16(h1_ref[i]))
                    acc = bf16_to_float(float_to_bf16(w * hi + acc))
                h2_ref.append(max(0.0, acc))

            exp_path = os.path.join(gpu_dir, 'data_ids_trained_expected.txt')
            with open(exp_path, 'w') as f:
                f.write("# Expected DMEM values after kernel execution\n")
                f.write("# Format: DMEM_ADDR HEX_VALUE FLOAT_VALUE\n")
                for n in range(16):
                    word = replicate_bf16(h1_ref[n])
                    f.write(f"203{'+' if n else ''} "
                            f"{n+203} {format_hex64(word)} {h1_ref[n]:.6f}\n"
                            if n else
                            f"{203} {format_hex64(word)} {h1_ref[0]:.6f}\n")
                for n in range(8):
                    word = replicate_bf16(h2_ref[n])
                    f.write(f"{355+n} {format_hex64(word)} {h2_ref[n]:.6f}\n")
                for n in range(2):
                    word = replicate_bf16(out[n])
                    f.write(f"{381+n} {format_hex64(word)} {out[n]:.6f}\n")
            #print(f"    Wrote expected to {exp_path}")

            # Also write expected hex for generic testbench
            exp_hex_path = os.path.join(gpu_dir, 'expected_ids_trained.hex')
            with open(exp_hex_path, 'w') as f:
                for v in h1_ref:
                    f.write(format_hex64(replicate_bf16(v)) + '\n')
                for v in h2_ref:
                    f.write(format_hex64(replicate_bf16(v)) + '\n')
                for v in out:
                    f.write(format_hex64(replicate_bf16(v)) + '\n')
            #print(f"    Wrote expected hex to {exp_hex_path}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Architecture:     {n_features} -> 16 -> 8 -> 2")
    print(f"  Selected features: {selected_names}")
    print(f"  BF16 test accuracy: {bf16_acc:.4f}")
    print(f"  Parameters:        {n_features*16+16 + 16*8+8 + 8*2+2} = "
          f"{n_features*16 + 16*8 + 8*2} weights + {16+8+2} biases")
    print(f"\nFiles generated:")
    print(f"  Weights:    programs/gpu/data_ids_trained_weights.json")
    print(f"  DMEM hex:   programs/gpu/data_ids_trained.hex")
    print(f"  Expected:   programs/gpu/expected_ids_trained.hex")
    #print(f"\nTo simulate:")
    #print(f"  cp programs/gpu/data_ids_trained.hex programs/gpu/data_ids_11_16_8_2.hex")
    #print(f"  cp programs/gpu/expected_ids_trained.hex programs/gpu/expected_ids.hex")
    #print(f"  iverilog -g2001 -I include -I src/gpu -o tb_ids src/gpu/*.v tb/tb_ids_generic.v")
    #print(f"  vvp tb_ids")


if __name__ == '__main__':
    main()
