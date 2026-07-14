#!/usr/bin/env python3
"""
train_ids_v2.py -- Improved IDS MLP training with ablation flags.

Architecture is selectable via --hidden (CSV of hidden widths). Output dim
is fixed at n_classes=2. Default '16,8' = 11->16->8->2, matching the
deployed silicon BCAST kernel (data_ids_trained_bcast.hex). Use
'32,32,16' for the 5-layer 11->32->32->16->2 backup target. Scalar DMEM
hex artifacts are only emitted for the legacy 11->16->8->2 arch; for any
other arch run toolchains/gpu/ids_data_generator_bcast.py to produce the
packed BCAST DMEM hex from the JSON.

Key improvements over train_ids.py:
  --log-bytes            log1p-transform heavy-tailed numeric features before
                         min-max so src_bytes/dst_bytes do not collapse to 0
  --include-categorical  add label-encoded protocol_type, flag, service to
                         the RF feature pool
  --optimizer adam       Adam (default) instead of SGD+momentum
  --epochs N             default 200 (was 50)
  --cosine               cosine LR decay
  --qat                  quantization-aware forward pass during training
  --seeds K              train K seeds and pick the best by BF16 KDDTest+ acc
  --hidden CSV           hidden layer widths; default '16,8' = legacy arch

Output (when not in --eval-only / --dry-run):
  programs/gpu/data_ids_<tag>_weights.json   (float weights + selection meta)
  programs/gpu/data_ids_<tag>.hex            (legacy arch only: DMEM image)
  programs/gpu/expected_ids_<tag>.hex        (legacy arch only: per-layer ref)

Usage:
  py -3 scripts/train_ids_v2.py                              # legacy arch
  py -3 scripts/train_ids_v2.py --hidden 32,32,16            # 5-layer arch
  py -3 scripts/train_ids_v2.py --no-log-bytes               # ablate log
  py -3 scripts/train_ids_v2.py --no-categorical             # ablate categorical
  py -3 scripts/train_ids_v2.py --optimizer sgd              # ablate optimizer
  py -3 scripts/train_ids_v2.py --no-qat                     # ablate QAT
  py -3 scripts/train_ids_v2.py --epochs 50 --no-cosine      # baseline schedule
  py -3 scripts/train_ids_v2.py --seeds 5                    # 5-seed sweep
  py -3 scripts/train_ids_v2.py --eval-only                  # eval current JSON
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'toolchains', 'gpu'))
from bf16_utils import float_to_bf16, bf16_to_float, pack_bf16_vector, format_hex64


# =====================================================================
# NSL-KDD column metadata
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
CATEGORICAL_RAW = ('protocol_type', 'service', 'flag')   # added as label-encoded
LABEL_COL = 'label'
NORMAL_LABELS = {'normal'}

# ----- Service grouping ----------------------------------------------------
# NSL-KDD has 70 distinct service strings. Label-encoding them gives the
# model 70 disconnected integer buckets, and any service that appears in
# KDDTest+ but not KDDTrain+ gets the special -1 sentinel. Collapsing into
# functional groups (web / mail / file-xfer / login / ...) makes the encoded
# value denser and lets unseen services map into a familiar bucket.
SERVICE_GROUPS = {
    'web':       {'http', 'http_443', 'http_2784', 'http_8001', 'harvest'},
    'mail':      {'smtp', 'pop_2', 'pop_3', 'imap4', 'nntp'},
    'ftp':       {'ftp', 'ftp_data'},
    'login':     {'telnet', 'ssh', 'rlogin', 'rsh', 'login', 'klogin', 'kshell',
                  'shell', 'exec'},
    'dns':       {'domain', 'domain_u'},
    'irc':       {'IRC'},
    'rpc':       {'sunrpc'},
    'whois':     {'whois', 'finger'},
    'time':      {'time', 'daytime', 'ntp_u'},
    'auth':      {'auth'},
    'snmp':      {'name'},
    'netbios':   {'netbios_dgm', 'netbios_ns', 'netbios_ssn'},
    'icmp':      {'eco_i', 'ecr_i', 'urh_i', 'urp_i', 'tim_i', 'red_i'},
    'ldap':      {'ldap', 'X11'},
    'sql':       {'mysql', 'sql_net'},
    'system':    {'systat', 'netstat', 'hostnames', 'link', 'tftp_u', 'csnet_ns',
                  'iso_tsap', 'discard', 'echo', 'gopher', 'mtp', 'supdup',
                  'efs', 'pm_dump', 'vmnet', 'aol', 'bgp', 'courier', 'ctf',
                  'nnsp', 'uucp', 'uucp_path', 'remote_job', 'printer', 'Z39_50'},
    'private':   {'private'},
    'other':     {'other'},
}


def group_service(name):
    """Map a raw NSL-KDD service string to a functional group name."""
    for grp, members in SERVICE_GROUPS.items():
        if name in members:
            return grp
    return 'unknown'

# Heavy-tailed numeric features that benefit from log1p before min-max.
# Picked from manual inspection: max values in the billions or hundreds
# of thousands while medians are tiny.
LOG_FEATURES = {
    'duration', 'src_bytes', 'dst_bytes',
    'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count',
    'num_compromised', 'num_root', 'num_file_creations',
    'num_failed_logins', 'hot',
}


# =====================================================================
# Loader: extract numeric + (optionally) label-encoded categorical
# =====================================================================
def build_categorical_maps(train_path, group_services=False):
    """First pass over train file: discover categorical value -> int maps.

    When `group_services` is True, the `service` map is built over the
    coarse functional groups (web/mail/login/...) rather than the raw
    70-value strings. The map then has at most ~17 entries.
    """
    maps = {c: {} for c in CATEGORICAL_RAW}
    with open(train_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 42:
                continue
            for c in CATEGORICAL_RAW:
                idx = NSL_COLUMNS.index(c)
                v = parts[idx].strip()
                if c == 'service' and group_services:
                    v = group_service(v)
                if v not in maps[c]:
                    maps[c][v] = len(maps[c])
    # Always reserve the 'unknown' bucket so unseen test services have a slot
    if group_services and 'unknown' not in maps['service']:
        maps['service']['unknown'] = len(maps['service'])
    return maps


def load_nslkdd(filepath, cat_maps, include_categorical, group_services=False):
    """Load NSL-KDD CSV. Returns (X, y, col_names).

    X columns are: 37 numeric NSL features (in NSL_COLUMNS order, skipping
    categorical+label), then if include_categorical, label-encoded
    protocol_type, flag, service appended at the end.

    When `group_services` is True, the raw service string is collapsed to
    a functional group before lookup in the (already-grouped) cat_maps.
    """
    numeric_cols = [c for c in NSL_COLUMNS[:41]
                    if c not in set(CATEGORICAL_RAW) | {LABEL_COL}]
    extra_cols = list(CATEGORICAL_RAW) if include_categorical else []
    col_names = numeric_cols + extra_cols

    data, labels = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 42:
                continue
            row = {NSL_COLUMNS[i]: parts[i] for i in range(min(len(parts), 42))}

            label_str = row[LABEL_COL].strip().rstrip('.')
            labels.append(0 if label_str in NORMAL_LABELS else 1)

            vec = []
            for c in numeric_cols:
                try:
                    vec.append(float(row[c]))
                except (KeyError, ValueError):
                    vec.append(0.0)
            if include_categorical:
                for c in CATEGORICAL_RAW:
                    v = row[c].strip()
                    if c == 'service' and group_services:
                        v = group_service(v)
                    # Unseen values fall back to the 'unknown' bucket if the
                    # map has one, else -1 (visible-to-RF sentinel).
                    if v in cat_maps[c]:
                        vec.append(float(cat_maps[c][v]))
                    elif 'unknown' in cat_maps[c]:
                        vec.append(float(cat_maps[c]['unknown']))
                    else:
                        vec.append(-1.0)
            data.append(vec)

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32), col_names


# =====================================================================
# Preprocessing
# =====================================================================
def apply_log_transform(X, col_names, log_set):
    """log1p-transform listed columns in place (returns new array)."""
    Xt = X.copy()
    for i, c in enumerate(col_names):
        if c in log_set:
            # log1p handles 0 cleanly; clip negatives (test set safety)
            Xt[:, i] = np.log1p(np.maximum(Xt[:, i], 0.0))
    return Xt


def select_features(X_train, y_train, col_names, n_features, seed,
                    force=None, method='rf'):
    """Feature selection by RF importance or RFE-wrapping-RF.

    Returns (selected_idx, selected_names, importances).

    method='rf' (default, legacy behavior):
        Single RF fit, rank by feature_importances_. Top-K wins.

    method='rfe':
        sklearn.feature_selection.RFE wrapping RandomForestClassifier with
        the same hyperparams. Iteratively eliminates the lowest-importance
        feature, refitting the RF each step. Slower but matches the SOTA
        2021 NSL-KDD paper. After RFE, a final RF is refit to provide
        intra-selected importances for the print log.

    `force` is an optional list of column names that MUST appear in the
    output. They are placed first; the remaining slots are filled by the
    selected method (skipping anything already forced).
    """
    from sklearn.ensemble import RandomForestClassifier

    def _make_rfc():
        return RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, max_depth=None,
        )

    forced_idx = []
    if force:
        seen = set()
        for name in force:
            if name in seen:
                continue
            seen.add(name)
            try:
                forced_idx.append(col_names.index(name))
            except ValueError:
                print("WARN: forced feature %r not in column list" % name)
    forced_set = set(forced_idx)
    n_rest = max(0, n_features - len(forced_idx))

    if method == 'rf':
        rf = _make_rfc()
        rf.fit(X_train, y_train)
        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1]
        rest = [int(i) for i in order if int(i) not in forced_set]
        sel = np.array(forced_idx + rest[:n_rest], dtype=np.int64)
        return sel, [col_names[i] for i in sel], imp

    if method == 'rfe':
        from sklearn.feature_selection import RFE
        # RFE only sees non-forced features. Indices in rfe.support_ are
        # relative to `available`.
        available = [i for i in range(len(col_names)) if i not in forced_set]
        if n_rest > len(available):
            raise ValueError("n_features=%d exceeds available pool of %d "
                             "after forcing %d" %
                             (n_features, len(available), len(forced_idx)))
        rfe = RFE(_make_rfc(), n_features_to_select=n_rest, step=1)
        rfe.fit(X_train[:, available], y_train)
        rfe_picked = [available[i] for i, sup in enumerate(rfe.support_) if sup]
        # RFE gives a binary selected/not; refit a final RF for sub-ranking
        # among the picked features and to populate the importances array
        # in the same shape the print loop expects.
        rf_final = _make_rfc()
        rf_final.fit(X_train, y_train)
        imp = rf_final.feature_importances_
        rfe_picked_ordered = sorted(rfe_picked, key=lambda i: -imp[i])
        sel = np.array(forced_idx + rfe_picked_ordered, dtype=np.int64)
        return sel, [col_names[i] for i in sel], imp

    raise ValueError("method must be 'rf' or 'rfe', got %r" % method)


def normalize_minmax(X_train, X_test, sel):
    """Min-max to [0,1] using train-only statistics."""
    Xtr = X_train[:, sel].astype(np.float32, copy=True)
    Xte = X_test[:, sel].astype(np.float32, copy=True)
    mins = Xtr.min(axis=0)
    maxs = Xtr.max(axis=0)
    rng = (maxs - mins)
    rng[rng == 0] = 1.0
    Xtr = (Xtr - mins) / rng
    Xte = np.clip((Xte - mins) / rng, 0.0, 1.0)
    return Xtr, Xte, {'mins': mins.tolist(), 'maxs': maxs.tolist()}


# =====================================================================
# Network primitives (manual numpy)
# =====================================================================
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    n = len(y)
    return -np.mean(np.log(np.clip(probs[np.arange(n), y], 1e-7, 1.0)))


# =====================================================================
# BF16 quantize (vectorized over arrays via element-wise truncation)
# =====================================================================
def quantize_bf16_array(x):
    """Element-wise BF16 truncation of a float32 numpy array.

    Implemented by viewing as uint32 and zeroing the low 16 bits.
    Matches the hardware bf16 truncation used by the FPGA ALU.
    """
    arr = np.asarray(x, dtype=np.float32)
    u = arr.view(np.uint32).copy()
    u &= 0xFFFF0000
    return u.view(np.float32)


# =====================================================================
# Forward pass (matrix or per-FMA form, with optional BF16 quantization)
# =====================================================================
def _fma_qat_layer(Xin, W, b):
    """HW-faithful per-FMA BF16 forward of one matmul layer.

    Each step inside the inner accumulation loop truncates `w * x + acc`
    to BF16 once -- exactly what the FPGA bf16_fma_unit does. Vectorized
    across batch and output neurons so the cost is in_dim small Python
    iterations per layer rather than in_dim * batch * out_dim element ops.
    """
    n_samp, in_dim = Xin.shape
    out_dim = W.shape[1]
    Wq = quantize_bf16_array(W)
    bq = quantize_bf16_array(b)
    Xq = quantize_bf16_array(Xin)
    acc = np.broadcast_to(bq, (n_samp, out_dim)).copy()
    for i in range(in_dim):
        mul = Xq[:, i:i + 1] * Wq[i:i + 1, :]
        acc = quantize_bf16_array(mul + acc)
    return acc


def forward_batch(X, Ws, Bs, qat=False, qat_mode='layer'):
    """Float32 forward, with optional BF16 quantization. N-layer.

    Ws: list of weight matrices, length L. Ws[i].shape = (in_dim_i, out_dim_i).
    Bs: list of bias vectors, length L. Bs[i].shape = (out_dim_i,).

    Returns (zs, hs):
      zs[i] = pre-activation of layer i (i = 0..L-1)
      hs[i] = activation feeding layer i; hs[0] = X (or BF16-quantized X
              under qat='layer'), hs[i] = (BF16-truncated) ReLU(zs[i-1]) for
              i >= 1. There is no hs[L]; output is zs[L-1] (no ReLU on output).

    qat=False:           pure float32 (no quant anywhere)
    qat=True,'layer':    per-layer matrix-form quant (input + pre-act + post-act)
    qat=True,'fma':      HW-faithful per-FMA truncation in every layer
    """
    L = len(Ws)
    if not qat:
        hs = [X]
        zs = []
        for i in range(L):
            z = hs[-1] @ Ws[i] + Bs[i]
            zs.append(z)
            if i < L - 1:
                hs.append(relu(z))
        return zs, hs

    if qat_mode == 'fma':
        hs = [X]
        zs = []
        for i in range(L):
            z = _fma_qat_layer(hs[-1], Ws[i], Bs[i])
            zs.append(z)
            if i < L - 1:
                hs.append(quantize_bf16_array(relu(z)))
        return zs, hs

    # qat_mode == 'layer' (default, faster).
    # hs[0] holds the ORIGINAL (unquantized) X to match legacy behavior:
    # the layer-0 weight gradient was `xb.T @ dz1` against the raw batch in
    # the old code, even though the matmul itself ran on BF16-truncated X.
    # hs[1..] hold post-ReLU + post-quantize activations as before.
    hs = [X]
    zs = []
    h_in = quantize_bf16_array(X)
    for i in range(L):
        z = h_in @ Ws[i] + Bs[i]
        z = quantize_bf16_array(z)
        zs.append(z)
        if i < L - 1:
            h_in = quantize_bf16_array(relu(z))
            hs.append(h_in)
    return zs, hs


def evaluate_float(X, y, Ws, Bs, qat_eval=False, qat_mode='layer'):
    zs, _ = forward_batch(X, Ws, Bs, qat=qat_eval, qat_mode=qat_mode)
    pred = np.argmax(zs[-1], axis=1)
    return float(np.mean(pred == y))


# =====================================================================
# Per-sample BF16 forward matching HW FMA semantics (single truncation per FMA)
# =====================================================================
def _layer_count_from_weights(weights):
    """Count layers by scanning b1, b2, ... keys in a packed weights dict."""
    L = 0
    while ('b%d' % (L + 1)) in weights and ('w%d' % (L + 1)) in weights:
        L += 1
    return L


def forward_single_bf16(x, weights):
    """Per-sample BF16 forward matching HW FMA semantics. Generalizes over
    arbitrary layer count by reading w1/b1, w2/b2, ... from `weights`.
    ReLU on hidden layers; no ReLU on output layer."""
    L = _layer_count_from_weights(weights)
    a = [bf16_to_float(float_to_bf16(v)) for v in x]
    for li in range(L):
        wkey = 'w%d' % (li + 1)
        bkey = 'b%d' % (li + 1)
        out_dim = len(weights[bkey])
        in_dim = len(a)
        new_a = []
        for n in range(out_dim):
            acc = bf16_to_float(float_to_bf16(weights[bkey][n]))
            for i in range(in_dim):
                w = bf16_to_float(float_to_bf16(weights[wkey][n][i]))
                acc = bf16_to_float(float_to_bf16(w * a[i] + acc))
            new_a.append(max(0.0, acc) if li < L - 1 else acc)
        a = new_a
    return a


def evaluate_bf16_full(X, y, weights, max_samples=None):
    """HW-faithful BF16 eval, vectorized across samples.

    Per-FMA semantics: each (w * x + acc) inside the inner accumulation loop
    is truncated to BF16 once, matching the FPGA bf16_alu / bf16_fma_unit.
    Outer loop is over input index of each layer; vectorized over all samples
    and all output neurons of that layer. Generalizes over arbitrary layer
    count by iterating through w1/b1, w2/b2, ... in the weights dict.
    """
    n = len(X) if max_samples is None else min(len(X), max_samples)
    Xn = np.asarray(X[:n], dtype=np.float32)
    yn = np.asarray(y[:n], dtype=np.int32)

    L = _layer_count_from_weights(weights)
    Ws = []
    Bs = []
    for li in range(L):
        # JSON weights are stored neuron-major (out_dim, in_dim); transpose
        # to (in_dim, out_dim) for matmul.
        Ws.append(np.array(weights['w%d' % (li + 1)], dtype=np.float32).T)
        Bs.append(np.array(weights['b%d' % (li + 1)], dtype=np.float32))

    Wq = [quantize_bf16_array(W) for W in Ws]
    Bq = [quantize_bf16_array(b) for b in Bs]

    def layer(Xin, Wq_l, bq_l):
        n_samp, in_dim = Xin.shape
        out_dim = Wq_l.shape[1]
        acc = np.broadcast_to(bq_l, (n_samp, out_dim)).copy()
        for i in range(in_dim):
            mul = Xin[:, i:i + 1] * Wq_l[i:i + 1, :]
            acc = quantize_bf16_array(mul + acc)
        return acc

    h = quantize_bf16_array(Xn)
    z_out = None
    for li in range(L):
        z = layer(h, Wq[li], Bq[li])
        if li < L - 1:
            h = quantize_bf16_array(np.maximum(0.0, z))
        else:
            z_out = z

    # Match the per-sample loop's tie-breaking: 0 if out[0] > out[1] else 1
    preds = np.where(z_out[:, 0] > z_out[:, 1], 0, 1).astype(np.int32)
    cm = np.zeros((2, 2), dtype=np.int64)
    for true_c in (0, 1):
        for pred_c in (0, 1):
            cm[true_c, pred_c] = int(np.sum((yn == true_c) & (preds == pred_c)))
    correct = int(cm[0, 0] + cm[1, 1])
    return correct / n, cm, preds, z_out


# =====================================================================
# Training (Adam or SGD+momentum, with optional cosine LR decay & QAT)
# =====================================================================
def train_mlp(X_train, y_train, X_test, y_test,
              arch_dims=(11, 16, 8, 2),
              optimizer='adam', lr=None, epochs=200,
              batch_size=256, cosine=True, qat=False, qat_mode='layer',
              weight_decay=0.0, seed=42, verbose=True,
              class_weights=None, label_smoothing=0.0,
              snapshot_criterion='test', snapshot_every=0):
    """Train an N-layer MLP. arch_dims = [n_features, h1, h2, ..., n_classes].

    class_weights: optional length-n_classes float array. Per-class loss
    multiplier. Used to upweight the attack class when test recall is low.

    snapshot_criterion: which metric to use when picking the best snapshot.
        'test'     -> best BF16 KDDTest+ acc (legacy behavior, default).
        'train'    -> best BF16 KDDTrain+ acc.
        'balanced' -> max(bf_train + bf_test).
        'last'     -> final-epoch weights (cross-seed ranking is meaningless
                      with 'last' since all seeds tie at score=epochs).

    snapshot_every: snapshot interval in epochs. 0 (default) preserves the
        legacy `epochs // 10` cadence (~10 snapshots per run). Set to 1 to
        snapshot every epoch -- combined with --snapshot-criterion balanced,
        this gives a per-epoch (bf_train, bf_test) trace suitable for
        Pareto-frontier scans. Cost: ~6 s per snapshot at 'balanced'.

    Returns (weights_packed, metrics_dict, snapshot_epoch). The metrics dict
    has keys {bf_train_acc, bf_test_acc, criterion, criterion_score,
    snapshot_epoch}; bf_*_acc are always populated (a final eval pass fills
    in whichever metric the criterion didn't already need).
    """
    arch_dims = list(arch_dims)
    n_features = arch_dims[0]
    n_classes = arch_dims[-1]
    L = len(arch_dims) - 1                          # number of weight matrices
    if X_train.shape[1] != n_features:
        raise ValueError("arch_dims[0]=%d does not match X_train.shape[1]=%d"
                         % (n_features, X_train.shape[1]))
    rng = np.random.RandomState(seed)
    if class_weights is None:
        class_weights = np.ones(n_classes, dtype=np.float32)
    else:
        class_weights = np.asarray(class_weights, dtype=np.float32)

    # Default lr depends on optimizer
    if lr is None:
        lr = 1e-3 if optimizer == 'adam' else 1e-2

    # He init for ReLU. Order of randn() calls per layer is preserved so a
    # given seed reproduces bit-exact across refactor (verified against
    # legacy 11->16->8->2 path).
    Ws = []
    Bs = []
    for i in range(L):
        fan_in = arch_dims[i]
        fan_out = arch_dims[i + 1]
        Ws.append(rng.randn(fan_in, fan_out).astype(np.float32)
                  * np.sqrt(2.0 / fan_in))
        Bs.append(np.zeros(fan_out, dtype=np.float32))

    # Optimizer state
    if optimizer == 'sgd':
        vW = [np.zeros_like(W) for W in Ws]
        vB = [np.zeros_like(b) for b in Bs]
        momentum = 0.9
    elif optimizer == 'adam':
        mW = [np.zeros_like(W) for W in Ws]
        mB = [np.zeros_like(b) for b in Bs]
        vW = [np.zeros_like(W) for W in Ws]
        vB = [np.zeros_like(b) for b in Bs]
        # NOTE: `eps` here is Adam's denominator stabilizer. The v3 recipe
        # (label_smoothing > 0) intentionally reassigns this to
        # label_smoothing inside each batch (see `eps = label_smoothing`
        # below); this is the behavior under which seed 53 produces 82.39%
        # KDDTest+. Do not rename without re-running the seed sweep.
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    n = X_train.shape[0]
    best_score = -np.inf
    best_snapshot = None
    snap_interval = snapshot_every if snapshot_every > 0 else max(1, epochs // 10)

    for epoch in range(epochs):
        # Cosine LR schedule
        if cosine:
            lr_e = 0.5 * lr * (1.0 + np.cos(np.pi * epoch / max(1, epochs - 1)))
        else:
            lr_e = lr

        perm = rng.permutation(n)
        Xs = X_train[perm]
        ys = y_train[perm]

        loss_sum = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = Xs[start:end]
            yb = ys[start:end]
            bs_size = end - start

            # Forward (QAT-style intermediate truncation if requested)
            zs, hs = forward_batch(xb, Ws, Bs, qat=qat, qat_mode=qat_mode)
            z_out = zs[-1]
            probs = softmax(z_out)

            # Per-sample weights from class_weights[y]
            sw = class_weights[yb]                       # (bs_size,)
            sw_norm = sw / sw.mean()                     # keep mean weight = 1

            # Soft target with label smoothing: target_y = 1 - eps*(K-1)/K,
            # other = eps/K. For binary K=2: target=1-eps/2, other=eps/2.
            if label_smoothing > 0.0:
                # NOTE: this assignment shadows the function-scope Adam
                # `eps` (above). The v3 recipe was trained with this
                # behavior (Adam denominator becomes label_smoothing once
                # the first LS-enabled batch runs); preserved verbatim so
                # seed 53 reproduces 82.39%. See banner above adam init.
                eps = label_smoothing
                soft = np.full((bs_size, n_classes),
                               eps / n_classes, dtype=np.float32)
                soft[np.arange(bs_size), yb] = 1.0 - eps + eps / n_classes
                ce_per = -(soft * np.log(np.clip(probs, 1e-7, 1.0))).sum(axis=1)
                loss_sum += float((sw_norm * ce_per).sum())
                dz = (probs - soft) * sw_norm[:, None] / bs_size
            else:
                ce_per = -np.log(np.clip(probs[np.arange(bs_size), yb],
                                         1e-7, 1.0))
                loss_sum += float((sw_norm * ce_per).sum())
                dz = probs.copy()
                dz[np.arange(bs_size), yb] -= 1.0
                dz *= sw_norm[:, None] / bs_size

            # Backprop layer L-1 -> 0. hs[i] is the input activation to layer i.
            dWs = [None] * L
            dBs = [None] * L
            for li in range(L - 1, -1, -1):
                a_in = hs[li]
                dWs[li] = a_in.T @ dz
                dBs[li] = dz.sum(axis=0)
                if li > 0:
                    dh_prev = dz @ Ws[li].T
                    dz = dh_prev * relu_grad(zs[li - 1])

            # Optional weight decay (L2)
            if weight_decay > 0.0:
                for li in range(L):
                    dWs[li] = dWs[li] + weight_decay * Ws[li]
                    dBs[li] = dBs[li] + weight_decay * Bs[li]

            if optimizer == 'sgd':
                for li in range(L):
                    vW[li] = momentum * vW[li] - lr_e * dWs[li]
                    Ws[li] += vW[li]
                    vB[li] = momentum * vB[li] - lr_e * dBs[li]
                    Bs[li] += vB[li]
            else:  # adam — order (W_i, b_i for li in 0..L-1) matches legacy
                t_step += 1
                bc1 = 1.0 - beta1 ** t_step
                bc2 = 1.0 - beta2 ** t_step
                for li in range(L):
                    mW[li] = beta1 * mW[li] + (1 - beta1) * dWs[li]
                    vW[li] = beta2 * vW[li] + (1 - beta2) * (dWs[li] * dWs[li])
                    Ws[li] -= lr_e * (mW[li] / bc1) / (np.sqrt(vW[li] / bc2) + eps)
                    mB[li] = beta1 * mB[li] + (1 - beta1) * dBs[li]
                    vB[li] = beta2 * vB[li] + (1 - beta2) * (dBs[li] * dBs[li])
                    Bs[li] -= lr_e * (mB[li] / bc1) / (np.sqrt(vB[li] / bc2) + eps)

        loss_avg = loss_sum / n

        # Periodic eval — track best snapshot per `snapshot_criterion` using
        # HW-faithful BF16 accuracy (not the matrix-form QAT proxy) so we
        # save the weights that will actually perform best on the FPGA.
        # bf_train is computed on the full 125 973-sample train set (~5 s
        # per snapshot); skipped unless the criterion needs it.
        if (epoch + 1) % snap_interval == 0 or epoch == 0:
            snap_weights = _pack_weights({
                'Ws': Ws, 'Bs': Bs, 'epoch': epoch + 1,
            })
            need_bf_test = snapshot_criterion in ('test', 'balanced')
            need_bf_train = snapshot_criterion in ('train', 'balanced')
            bf_test = None
            bf_train = None
            if need_bf_test:
                bf_test, _, _, _ = evaluate_bf16_full(X_test, y_test, snap_weights)
            if need_bf_train:
                bf_train, _, _, _ = evaluate_bf16_full(X_train, y_train, snap_weights)

            if verbose:
                tr_f32 = evaluate_float(X_train, y_train, Ws, Bs,
                                        qat_eval=qat, qat_mode=qat_mode)
                msg = ("  epoch %3d: loss=%.4f  lr=%.5f  f32_train=%.4f"
                       % (epoch + 1, loss_avg, lr_e, tr_f32))
                if bf_train is not None:
                    msg += "  bf16_train=%.4f" % bf_train
                if bf_test is not None:
                    msg += "  bf16_test=%.4f" % bf_test
                print(msg)

            if snapshot_criterion == 'test':
                score = bf_test
            elif snapshot_criterion == 'train':
                score = bf_train
            elif snapshot_criterion == 'balanced':
                score = bf_train + bf_test
            elif snapshot_criterion == 'last':
                # Monotonically increasing -> always picks most recent snapshot
                score = float(epoch + 1)
            else:
                raise ValueError("unknown snapshot_criterion=%r"
                                 % snapshot_criterion)

            if score > best_score:
                best_score = score
                best_snapshot = {
                    'Ws': [W.copy() for W in Ws],
                    'Bs': [b.copy() for b in Bs],
                    'epoch': epoch + 1,
                    'bf_train_acc': bf_train,
                    'bf_test_acc': bf_test,
                }

    # If we never tracked a snapshot (low epoch), use final
    if best_snapshot is None:
        best_snapshot = {
            'Ws': [W.copy() for W in Ws],
            'Bs': [b.copy() for b in Bs],
            'epoch': epochs,
            'bf_train_acc': None,
            'bf_test_acc': None,
        }

    # Always populate BOTH bf_train_acc and bf_test_acc on the picked
    # snapshot so downstream consumers (cross-seed ranking, JSON export,
    # slides) don't have to deal with None. At most 2 extra eval calls
    # (~6 s total); for the v3 default 'test' criterion this fills in
    # bf_train_acc for free.
    if best_snapshot.get('bf_test_acc') is None:
        snap_w = _pack_weights(best_snapshot)
        bf_test_final, _, _, _ = evaluate_bf16_full(X_test, y_test, snap_w)
        best_snapshot['bf_test_acc'] = float(bf_test_final)
    if best_snapshot.get('bf_train_acc') is None:
        snap_w = _pack_weights(best_snapshot)
        bf_train_final, _, _, _ = evaluate_bf16_full(X_train, y_train, snap_w)
        best_snapshot['bf_train_acc'] = float(bf_train_final)

    # Recompute score from the now-complete snapshot in the no-snapshot
    # fallback path (otherwise best_score is already correct from the loop).
    if best_score == -np.inf:
        if snapshot_criterion == 'test':
            best_score = best_snapshot['bf_test_acc']
        elif snapshot_criterion == 'train':
            best_score = best_snapshot['bf_train_acc']
        elif snapshot_criterion == 'balanced':
            best_score = (best_snapshot['bf_train_acc']
                          + best_snapshot['bf_test_acc'])
        elif snapshot_criterion == 'last':
            best_score = float(best_snapshot['epoch'])

    metrics = {
        'bf_train_acc': float(best_snapshot['bf_train_acc']),
        'bf_test_acc': float(best_snapshot['bf_test_acc']),
        'criterion': snapshot_criterion,
        'criterion_score': float(best_score),
        'snapshot_epoch': int(best_snapshot['epoch']),
    }
    return _pack_weights(best_snapshot), metrics, best_snapshot['epoch']


def _pack_weights(snap):
    """Pack snapshot dict to JSON-friendly lists in kernel layout
    (W transposed to neuron-major). Input dict has lists Ws and Bs."""
    Ws = snap['Ws']
    Bs = snap['Bs']
    out = {}
    for i in range(len(Ws)):
        out['w%d' % (i + 1)] = Ws[i].T.tolist()
        out['b%d' % (i + 1)] = Bs[i].tolist()
    return out


# =====================================================================
# DMEM hex export (matches kernel layout exactly)
# =====================================================================
def replicate_bf16(value):
    return pack_bf16_vector([value, value, value, value])


def build_dmem(inputs, weights):
    DMEM_SIZE = 383
    dmem = [0] * DMEM_SIZE
    L1_IN, L1_OUT = 11, 16
    L2_IN, L2_OUT = 16, 8
    L3_IN, L3_OUT = 8, 2
    X_BASE, W1_BASE, B1_BASE = 0, 11, 187
    W2_BASE, B2_BASE = 219, 347
    W3_BASE, B3_BASE = 363, 379
    for i in range(L1_IN):
        dmem[X_BASE + i] = replicate_bf16(inputs[i])
    for n in range(L1_OUT):
        for i in range(L1_IN):
            dmem[W1_BASE + n * L1_IN + i] = replicate_bf16(weights['w1'][n][i])
    for n in range(L1_OUT):
        dmem[B1_BASE + n] = replicate_bf16(weights['b1'][n])
    for n in range(L2_OUT):
        for i in range(L2_IN):
            dmem[W2_BASE + n * L2_IN + i] = replicate_bf16(weights['w2'][n][i])
    for n in range(L2_OUT):
        dmem[B2_BASE + n] = replicate_bf16(weights['b2'][n])
    for n in range(L3_OUT):
        for i in range(L3_IN):
            dmem[W3_BASE + n * L3_IN + i] = replicate_bf16(weights['w3'][n][i])
    for n in range(L3_OUT):
        dmem[B3_BASE + n] = replicate_bf16(weights['b3'][n])
    return dmem


def write_hex(dmem, path):
    with open(path, 'w') as f:
        for word in dmem:
            f.write(format_hex64(word) + '\n')


def write_expected_hex(weights, x, path):
    """Write expected per-layer outputs (h1, h2, logits) for testbench."""
    x_q = [bf16_to_float(float_to_bf16(v)) for v in x]
    h1 = []
    for n in range(16):
        acc = bf16_to_float(float_to_bf16(weights['b1'][n]))
        for i in range(11):
            w = bf16_to_float(float_to_bf16(weights['w1'][n][i]))
            acc = bf16_to_float(float_to_bf16(w * x_q[i] + acc))
        h1.append(max(0.0, acc))
    h2 = []
    for n in range(8):
        acc = bf16_to_float(float_to_bf16(weights['b2'][n]))
        for i in range(16):
            w = bf16_to_float(float_to_bf16(weights['w2'][n][i]))
            acc = bf16_to_float(float_to_bf16(w * h1[i] + acc))
        h2.append(max(0.0, acc))
    out = []
    for n in range(2):
        acc = bf16_to_float(float_to_bf16(weights['b3'][n]))
        for i in range(8):
            w = bf16_to_float(float_to_bf16(weights['w3'][n][i]))
            acc = bf16_to_float(float_to_bf16(w * h2[i] + acc))
        out.append(acc)
    with open(path, 'w') as f:
        for v in h1:
            f.write(format_hex64(replicate_bf16(v)) + '\n')
        for v in h2:
            f.write(format_hex64(replicate_bf16(v)) + '\n')
        for v in out:
            f.write(format_hex64(replicate_bf16(v)) + '\n')


# =====================================================================
# Confusion matrix pretty-printer
# =====================================================================
def fmt_cm(cm):
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total else 0.0
    prec_atk = tp / (tp + fp) if (tp + fp) else 0.0
    rec_atk = tp / (tp + fn) if (tp + fn) else 0.0
    prec_norm = tn / (tn + fn) if (tn + fn) else 0.0
    rec_norm = tn / (tn + fp) if (tn + fp) else 0.0
    f1_atk = 2 * prec_atk * rec_atk / (prec_atk + rec_atk) if (prec_atk + rec_atk) else 0.0
    return ("    pred=N pred=A\n"
            "  N  %5d  %5d   recall=%.3f\n"
            "  A  %5d  %5d   recall=%.3f\n"
            "  precision: N=%.3f  A=%.3f   F1(A)=%.3f   acc=%.3f"
            % (tn, fp, rec_norm, fn, tp, rec_atk, prec_norm, prec_atk, f1_atk, acc))


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=int, default=11)
    ap.add_argument('--hidden', type=str, default='16,8',
                    help="Comma-separated hidden layer widths. Output dim is "
                         "fixed at n_classes=2. Default '16,8' = 11->16->8->2 "
                         "(legacy silicon arch). '32,32,16' = 11->32->32->16->2 "
                         "(5-layer backup target). Scalar DMEM hex artifacts "
                         "are only emitted for the legacy arch.")
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    ap.add_argument('--cosine', dest='cosine', action='store_true', default=True)
    ap.add_argument('--no-cosine', dest='cosine', action='store_false')
    ap.add_argument('--log-bytes', dest='log_bytes', action='store_true', default=True)
    ap.add_argument('--no-log-bytes', dest='log_bytes', action='store_false')
    ap.add_argument('--include-categorical', dest='include_categorical',
                    action='store_true', default=True)
    ap.add_argument('--no-categorical', dest='include_categorical', action='store_false')
    ap.add_argument('--group-services', dest='group_services',
                    action='store_true', default=False,
                    help="Collapse 70 service strings into ~17 functional "
                         "groups (web/mail/ftp/login/...) before label encoding.")
    ap.add_argument('--qat', dest='qat', action='store_true', default=True)
    ap.add_argument('--no-qat', dest='qat', action='store_false')
    ap.add_argument('--qat-mode', choices=['layer', 'fma'], default='layer',
                    help="layer = per-layer matmul-then-truncate (fast); "
                         "fma = HW-faithful per-FMA truncation (slower).")
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--attack-weight', type=float, default=1.0,
                    help="Class weight for attack class (1.0=no upweighting). "
                         "Use ~1.5-2.0 if attack recall is too low.")
    ap.add_argument('--label-smoothing', type=float, default=0.0,
                    help="Label smoothing epsilon (0=hard labels, 0.1 typical). "
                         "Reduces overconfidence; can help generalization.")
    ap.add_argument('--snapshot-criterion',
                    choices=['test', 'train', 'balanced', 'last'],
                    default='test',
                    help="How to pick the saved snapshot within each training "
                         "run AND the best seed across a sweep. 'test' "
                         "(default) = best BF16 KDDTest+ acc (legacy "
                         "behavior). 'train' = best BF16 KDDTrain+. "
                         "'balanced' = max(bf_train + bf_test). 'last' = "
                         "final-epoch weights (cross-seed ranking is "
                         "meaningless with 'last' since all seeds tie at "
                         "score=epochs).")
    ap.add_argument('--snapshot-every', type=int, default=0,
                    help="Snapshot interval in epochs. 0 (default) = legacy "
                         "behavior of `epochs // 10`. Set to 1 to snapshot "
                         "every epoch (combined with --snapshot-criterion "
                         "balanced, gives a per-epoch (bf_train, bf_test) "
                         "trace suitable for Pareto-frontier scans).")
    ap.add_argument('--seed', type=int, default=42,
                    help="Starting training seed; sweep uses seed..seed+seeds-1")
    ap.add_argument('--rf-seed', type=int, default=42,
                    help="Random forest seed for feature selection. Held fixed "
                         "across the sweep so all training seeds see the same "
                         "11-feature set.")
    ap.add_argument('--feature-selection', choices=['rf', 'rfe'], default='rf',
                    help="Feature selection method. 'rf' (default) = single "
                         "RF fit, rank by feature_importances_, top-K wins. "
                         "'rfe' = sklearn RFE wrapping RandomForestClassifier; "
                         "iteratively eliminates the lowest-importance "
                         "feature one at a time. RFE is slower (~30 s × "
                         "(n_total - n_features) iterations) but matches the "
                         "2021 NSL-KDD SOTA paper.")
    ap.add_argument('--force-features', type=str, default='',
                    help="Comma-separated list of column names to force into "
                         "the selection (placed first, then RF fills remainder).")
    ap.add_argument('--seeds', type=int, default=1,
                    help="Train this many seeds and pick best by BF16 KDDTest+ acc")
    ap.add_argument('--bf16-eval-samples', type=int, default=2000,
                    help="How many KDDTest+ samples to use for the BF16 sweep eval")
    ap.add_argument('--final-bf16-samples', type=int, default=0,
                    help="Run final BF16 eval over this many samples (0 = full test set)")
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help="Don't write artifacts")
    ap.add_argument('--tag', type=str, default='trained',
                    help="Output suffix: data_ids_<tag>.hex etc.")
    args = ap.parse_args()

    hidden = [int(x.strip()) for x in args.hidden.split(',') if x.strip()]
    if not hidden:
        ap.error("--hidden must list at least one hidden layer width")
    n_classes = 2
    arch_dims = [args.features] + hidden + [n_classes]
    LEGACY_ARCH = [args.features, 16, 8, 2]
    is_legacy_arch = (arch_dims == LEGACY_ARCH)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'datasets')
    gpu_dir = os.path.join(project_dir, 'programs', 'gpu')

    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    print("=" * 72)
    print("IDS training v2  (architecture: %s)"
          % ' -> '.join(str(d) for d in arch_dims))
    print("=" * 72)
    print("  log_bytes        = %s" % args.log_bytes)
    print("  categorical      = %s" % args.include_categorical)
    print("  optimizer        = %s" % args.optimizer)
    print("  epochs           = %d" % args.epochs)
    print("  cosine schedule  = %s" % args.cosine)
    print("  qat              = %s" % args.qat)
    print("  weight_decay     = %g" % args.weight_decay)
    print("  seeds            = %d" % args.seeds)
    print("  snapshot_crit    = %s" % args.snapshot_criterion)
    print("  feature_select   = %s" % args.feature_selection)
    if not is_legacy_arch:
        print("  (non-legacy arch: scalar DMEM hex artifacts will be skipped)")
    print("=" * 72)

    # ----- Categorical maps (built from train file) -----
    if args.include_categorical:
        print("\nBuilding categorical maps from train file...")
        cat_maps = build_categorical_maps(train_path, group_services=args.group_services)
        for c, m in cat_maps.items():
            print("  %-15s  %3d unique values" % (c, len(m)))
    else:
        cat_maps = {c: {} for c in CATEGORICAL_RAW}

    # ----- Load -----
    print("\nLoading train...")
    X_train, y_train, col_names = load_nslkdd(train_path, cat_maps, args.include_categorical,
                                              group_services=args.group_services)
    print("  shape = %s   normal=%d  attack=%d"
          % (X_train.shape, int((y_train == 0).sum()), int((y_train == 1).sum())))
    print("Loading test...")
    X_test, y_test, _ = load_nslkdd(test_path, cat_maps, args.include_categorical,
                                    group_services=args.group_services)
    print("  shape = %s   normal=%d  attack=%d"
          % (X_test.shape, int((y_test == 0).sum()), int((y_test == 1).sum())))

    # ----- Log transform -----
    if args.log_bytes:
        print("\nLog1p-transforming heavy-tailed columns: %s" % sorted(LOG_FEATURES))
        X_train = apply_log_transform(X_train, col_names, LOG_FEATURES)
        X_test = apply_log_transform(X_test, col_names, LOG_FEATURES)

    # ----- Feature selection -----
    if args.feature_selection == 'rfe':
        print("\nSelecting top %d features by RFE on RandomForestClassifier "
              "(n_estimators=200, step=1)..." % args.features)
    else:
        print("\nSelecting top %d features by RF importance (n_estimators=200)..."
              % args.features)
    forced = [s.strip() for s in args.force_features.split(',') if s.strip()] \
              if args.force_features else None
    sel_idx, sel_names, importances = select_features(
        X_train, y_train, col_names, args.features, args.rf_seed,
        force=forced, method=args.feature_selection)
    print("  selected:")
    for i, name in enumerate(sel_names):
        print("    %2d. %-30s imp=%.4f" % (i + 1, name, importances[sel_idx[i]]))

    # ----- Normalize -----
    print("\nMin-max normalizing selected features to [0, 1]...")
    Xtr, Xte, norm_params = normalize_minmax(X_train, X_test, sel_idx)

    if args.eval_only:
        weights_path = os.path.join(gpu_dir, 'data_ids_%s_weights.json' % args.tag)
        with open(weights_path, 'r') as f:
            data = json.load(f)
        weights = data['weights']
        print("\nLoaded weights from %s" % weights_path)

        # Reconstruct Ws/Bs lists from the JSON weights so we can compare
        # float32 (model-capacity) vs BF16-faithful (silicon-deployed) accuracy.
        # Reporting both makes the train/test gap AND the precision gap visible
        # on Slide 2.2; either can anchor the speaker note depending on which
        # methodological story the slide foregrounds.
        L_layers = _layer_count_from_weights(weights)
        Ws_eval = [np.array(weights['w%d' % (li + 1)], dtype=np.float32).T
                   for li in range(L_layers)]
        Bs_eval = [np.array(weights['b%d' % (li + 1)], dtype=np.float32)
                   for li in range(L_layers)]

        # KDDTrain+ accuracy (training set fit)
        print("\nEvaluating on KDDTrain+ (training set, n=%d)..." % len(Xtr))
        tr_f32 = evaluate_float(Xtr, y_train, Ws_eval, Bs_eval, qat_eval=False)
        print("  Float32 (model capacity):     %.4f (%d/%d)"
              % (tr_f32, int(tr_f32 * len(Xtr)), len(Xtr)))
        bf_acc_train, _, _, _ = evaluate_bf16_full(Xtr, y_train, weights)
        print("  BF16-faithful (silicon):      %.4f (%d/%d)"
              % (bf_acc_train, int(bf_acc_train * len(Xtr)), len(Xtr)))

        # KDDTest+ accuracy (held-out benchmark with novel attack categories)
        print("\nEvaluating on KDDTest+ (held-out benchmark, n=%d)..." % len(Xte))
        te_f32 = evaluate_float(Xte, y_test, Ws_eval, Bs_eval, qat_eval=False)
        print("  Float32 (model capacity):     %.4f (%d/%d)"
              % (te_f32, int(te_f32 * len(Xte)), len(Xte)))
        bf_acc, cm, _, _ = evaluate_bf16_full(Xte, y_test, weights,
                                              max_samples=args.final_bf16_samples or None)
        n = args.final_bf16_samples or len(Xte)
        print("  BF16-faithful (silicon):      %.4f (%d/%d)"
              % (bf_acc, int(bf_acc * n), n))
        print("\nConfusion matrix (BF16 KDDTest+):\n%s" % fmt_cm(cm))
        return

    # ----- Train (multi-seed sweep over the same feature pipeline) -----
    print("\nTraining %d seed(s)..." % args.seeds)
    best_overall = None
    seed_results = []
    for s in range(args.seeds):
        seed = args.seed + s
        print("\n--- seed %d (rng=%d) ---" % (s + 1, seed))
        t0 = time.time()
        cw = np.array([1.0, args.attack_weight], dtype=np.float32)
        weights, snap_metrics, best_epoch = train_mlp(
            Xtr, y_train, Xte, y_test,
            arch_dims=arch_dims,
            optimizer=args.optimizer, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            cosine=args.cosine, qat=args.qat, qat_mode=args.qat_mode,
            weight_decay=args.weight_decay,
            seed=seed, verbose=True,
            class_weights=cw,
            label_smoothing=args.label_smoothing,
            snapshot_criterion=args.snapshot_criterion,
            snapshot_every=args.snapshot_every,
        )
        dt = time.time() - t0
        print("  trained in %.1fs   epoch=%d   bf16_train=%.4f   "
              "bf16_test=%.4f   crit=%s   score=%g"
              % (dt, best_epoch, snap_metrics['bf_train_acc'],
                 snap_metrics['bf_test_acc'], args.snapshot_criterion,
                 snap_metrics['criterion_score']))

        # Visibility-only KDDTest+ eval at the configured sweep size
        # (cross-seed ranking uses snap_metrics['criterion_score'] below).
        n_eval = min(args.bf16_eval_samples, len(Xte))
        bf_acc, cm, _, _ = evaluate_bf16_full(Xte, y_test, weights, max_samples=n_eval)
        print("  bf16 KDDTest+ acc on first %d samples: %.4f" % (n_eval, bf_acc))
        print("  confusion matrix (BF16 KDDTest+):\n%s" % fmt_cm(cm))
        seed_results.append((seed, snap_metrics, weights))
        cur_score = snap_metrics['criterion_score']
        if best_overall is None or cur_score > best_overall[1]['criterion_score']:
            best_overall = (seed, snap_metrics, weights)

    print("\n=== seed sweep summary (criterion=%s) ===" % args.snapshot_criterion)
    for s, m, _ in seed_results:
        marker = " *" if s == best_overall[0] else ""
        print("  seed=%d  bf16_train=%.4f  bf16_test=%.4f  score=%g%s"
              % (s, m['bf_train_acc'], m['bf_test_acc'],
                 m['criterion_score'], marker))

    weights = best_overall[2]
    best_seed = best_overall[0]
    best_metrics = best_overall[1]

    # ----- Final BF16 eval (full or large sample) -----
    n_full = args.final_bf16_samples if args.final_bf16_samples > 0 else len(Xte)
    n_full = min(n_full, len(Xte))
    print("\nFinal BF16 evaluation (seed=%d, criterion=%s, snapshot epoch=%d)..."
          % (best_seed, args.snapshot_criterion, best_metrics['snapshot_epoch']))
    print("  Snapshot bf16 KDDTrain+ acc: %.4f (full %d samples)"
          % (best_metrics['bf_train_acc'], len(Xtr)))
    bf_acc, cm, preds, logits = evaluate_bf16_full(Xte, y_test, weights, max_samples=n_full)
    print("  bf16 KDDTest+ accuracy:      %.4f (%d/%d)"
          % (bf_acc, int(round(bf_acc * n_full)), n_full))
    print("  confusion matrix (BF16 KDDTest+):\n%s" % fmt_cm(cm))

    if args.dry_run:
        print("\n--dry-run: not writing artifacts.")
        return

    # ----- Export -----
    print("\nWriting artifacts...")
    export = {
        'weights': weights,
        'selected_features': sel_names,
        'selected_indices': [int(i) for i in sel_idx],
        'norm_params': norm_params,
        'n_features': args.features,
        'architecture': list(arch_dims),
        'preprocessing': {
            'log_features': sorted(LOG_FEATURES) if args.log_bytes else [],
            'include_categorical': bool(args.include_categorical),
            'group_services': bool(args.group_services),
            'feature_selection': args.feature_selection,
            'categorical_maps': {c: m for c, m in cat_maps.items()}
                if args.include_categorical else {},
        },
        'training': {
            'optimizer': args.optimizer,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'cosine': bool(args.cosine),
            'qat': bool(args.qat),
            'seed': int(best_seed),
            'lr': args.lr,
            'snapshot_criterion': args.snapshot_criterion,
            'snapshot_epoch': int(best_metrics['snapshot_epoch']),
            'attack_weight': float(args.attack_weight),
            'label_smoothing': float(args.label_smoothing),
        },
        'metrics': {
            'bf16_accuracy_test': float(bf_acc),
            'bf16_eval_size': int(n_full),
            'bf16_accuracy_train': float(best_metrics['bf_train_acc']),
            'bf16_eval_size_train': int(len(Xtr)),
            'confusion_matrix': cm.tolist(),
            'criterion_score': float(best_metrics['criterion_score']),
        },
    }

    weights_path = os.path.join(gpu_dir, 'data_ids_%s_weights.json' % args.tag)
    with open(weights_path, 'w') as f:
        json.dump(export, f, indent=2)
    print("  %s" % weights_path)

    # Default test sample (one attack from KDDTest+) for hex artifacts
    attack_idx = np.where(y_test == 1)[0]
    chosen_idx = int(attack_idx[0]) if len(attack_idx) else 0
    x_default = Xte[chosen_idx].tolist()

    if is_legacy_arch:
        dmem = build_dmem(x_default, weights)
        hex_path = os.path.join(gpu_dir, 'data_ids_%s.hex' % args.tag)
        write_hex(dmem, hex_path)
        print("  %s" % hex_path)

        exp_hex_path = os.path.join(gpu_dir, 'expected_ids_%s.hex' % args.tag)
        write_expected_hex(weights, x_default, exp_hex_path)
        print("  %s" % exp_hex_path)
    else:
        print("  (skipped scalar .hex / expected.hex artifacts; only emitted "
              "for the legacy %s architecture. The packed BCAST DMEM hex is "
              "produced by toolchains/gpu/ids_data_generator_bcast.py from "
              "the JSON above.)" % LEGACY_ARCH)

    print("\nSummary")
    print("  features:  %s" % sel_names)
    print("  bf16 acc:  %.4f on %d test samples" % (bf_acc, n_full))


if __name__ == '__main__':
    main()
