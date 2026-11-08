#!/usr/bin/env python3
"""
Testbench data for the GCN + JumpingKnowledge backbone + pragma MLPs.

Emits tb_data/ from the REAL finetune_dataset graph and the REAL trained
weights of ONE checkpoint (GCN and pragma MLP alike, for one --expert), plus
two golden references computed in float64 with numpy:

    golden_jk.bin      GCN+JK output         -- checks stage 1
    golden_pragma.bin  pragma MLP output     -- checks stage 2 (final output)

WHAT THE GOLDEN IS
------------------
The reference matches the SEMANTICS OF THE KERNEL, not HierarchicalMoE.
model.cpp does plain GCN with self-loops, ReLU after every layer, and
JumpingKnowledge('max') over all six layer outputs:

    neighbour j -> i :  1 / sqrt((d_i + 1)(d_j + 1))
    self-loop        :  1 / (d_i + 1)
    out              :  max over layers 0..5 of relu(agg @ W.T + b)

which is PyG's gcn_norm with add_self_loops=True and no edge_weight, followed
by JumpingKnowledge(mode='max'). It deliberately does NOT include:

    * lin_edge   -- the Linear(139->1)+sigmoid edge weights. The kernel has no
                    edge features, so every edge weight is 1.0.
    * ELU        -- the kernel uses ReLU.
    * anything downstream of the pragma MLP (gnn7, pooling, output MLP).

Generating the golden from HierarchicalMoE's real forward pass instead would
fail for reasons that have nothing to do with the HLS.

The pragma-MLP golden is the author's Net.apply_pragam_as_MLP
(model.py:425-463, pragma_order='parallel_and_merge') applied to that JK
output, masks included -- see pragma_reference(). validate_pragma_reference.py
checks this function against the author's real model.

PADDING
-------
Node features are 82 wide, zero-padded to 96: lib's `linear` static_asserts
in_size % P_IN == 0, and 82 = 2 x 41, so P_IN=16 needs a multiple of 16.
The layer-0 weight matrix is padded identically, so the result is unchanged.

Layers 1-5 use 64-wide embeddings, but every weight port on the top is
[64][96] (one shared `linear` instance sized to the maximum). Their [64][64]
matrices are therefore zero-padded to [64][96]. Columns 64..95 are never read
at those layers -- `linear` loops j < actual_in_size = 64 -- so the padding is
inert.

MULTI-EDGES
-----------
The normalised adjacency is accumulated per edge, not per (src,dst) pair, so a
repeated edge contributes twice -- exactly like the kernel's neighbour table,
which holds one entry per edge.

Usage
    cd ~/gcn_jk_mlp
    source ~/fpga_accel/venv/bin/activate
    python dump_tb_data.py                  # largest graph in the dataset
    python dump_tb_data.py --graph syr2k    # a named kernel
    python dump_tb_data.py --expert 1       # another expert (GCN + pragma MLP)
    python dump_tb_data.py --list           # what's available
"""
import argparse
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tb_data")

HMOE_ROOT = "/home/rizwanpathan/hierarchical_moe/HierarchicalMoE"

IN_DIM_REAL = 82      # finetune_dataset node feature width
IN_DIM_PAD = 96       # what the kernel expects
D = 64                # embedding width
N_LAYERS = 6
N_MAX = 555           # kernel's fixed bounds
E_MAX = 1965


def find_dataset_dir():
    hits = sorted(glob.glob(os.path.join(HMOE_ROOT, "save/programl/*/finetune_dataset")))
    if not hits:
        raise FileNotFoundError("no finetune_dataset under save/programl")
    return hits[0]


def find_checkpoint():
    hits = sorted(glob.glob(os.path.join(
        HMOE_ROOT, "src/logs/**/*gnn-gcn/run1/1000_val_model_state_dict.pth"),
        recursive=True))
    if not hits:
        raise FileNotFoundError("no GCN checkpoint under src/logs")
    return hits[0]


def load_all(ddir):
    """Every sample, with its kernel name. Candidates of one kernel share
    bit-identical x / edge_index, so any one stands for the whole kernel as far
    as the GCN backbone is concerned."""
    out = []
    for f in sorted(glob.glob(os.path.join(ddir, "data_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append((f, d))
    return out


def build_reference(x_pad, edge_index, weights_pad, biases):
    """Six GCN layers with self-loops + ReLU, then element-wise max (JK).

    Dense float64 throughout. N <= 555 so a dense [N, N] adjacency is cheap and
    removes any ambiguity about neighbour ordering.
    """
    N = x_pad.shape[0]
    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    # in-degree excluding self-loops -- matches in_degree_table in the kernel
    indeg = np.bincount(dst, minlength=N).astype(np.float64)

    # normalised adjacency, accumulated per edge so multi-edges count twice
    A = np.zeros((N, N), dtype=np.float64)
    w_edge = 1.0 / np.sqrt((indeg[dst] + 1.0) * (indeg[src] + 1.0))
    np.add.at(A, (dst, src), w_edge)
    A[np.arange(N), np.arange(N)] += 1.0 / (indeg + 1.0)   # self-loop

    h = x_pad.astype(np.float64)              # layer 0 input is 96 wide
    jk = None
    per_layer = []
    for lyr in range(N_LAYERS):
        # Layer 0 consumes all 96 padded columns; layers 1-5 read only the
        # first 64, exactly as `linear` does with actual_in_size = 64.
        in_w = IN_DIM_PAD if lyr == 0 else D
        W = weights_pad[lyr][:, :in_w].astype(np.float64)  # (64, in_w)
        b = biases[lyr].astype(np.float64)                 # (64,)

        agg = A @ h[:, :in_w]
        h = np.maximum(agg @ W.T + b, 0.0)                 # relu
        per_layer.append(h.copy())
        jk = h.copy() if jk is None else np.maximum(jk, h)

    return jk.astype(np.float32), per_layer, indeg


def pragma_reference(jk, x_pragma, scope, sd, expert):
    """Net.apply_pragam_as_MLP, pragma_order='parallel_and_merge' (model.py:425).

    MLP.forward (utils.py:538): activation (ELU) after every layer but the last.
    Each per-pragma MLP sees mask_emb(cat(out, option), scope) and its output is
    masked by scope again; merge sees the masked concat and the result is
    out*(1-scope) + merge_out*scope.
    """
    def mlp(x, name):
        pre = f"experts.{expert}.MLPs_per_pragma.{name}.layers"
        W0 = sd[f"{pre}.0.weight"].numpy().astype(np.float64)
        b0 = sd[f"{pre}.0.bias"].numpy().astype(np.float64)
        W1 = sd[f"{pre}.1.weight"].numpy().astype(np.float64)
        b1 = sd[f"{pre}.1.bias"].numpy().astype(np.float64)
        h = x @ W0.T + b0
        h = np.where(h > 0, h, np.expm1(h))                # ELU, alpha=1
        return h @ W1.T + b1

    out = jk.astype(np.float64)
    xp = x_pragma.astype(np.float64)
    s = scope.astype(np.float64)[:, None]
    options = {"tile": xp[:, 0:1], "pipeline": xp[:, 1:2], "parallel": xp[:, 2:4]}

    in_merge = []
    for name in ("tile", "pipeline", "parallel"):
        mlp_inp = np.concatenate([out, options[name]], axis=1)
        in_merge.append(mlp(mlp_inp * s, name) * s)
    in_merge = np.concatenate(in_merge, axis=1)
    merged = mlp(in_merge * s, "merge")
    return (out * (1.0 - s) + merged * s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None,
                    help="kernel name (substring), e.g. syr2k. Default: largest graph.")
    ap.add_argument("--expert", type=int, default=0,
                    help="which HMoE expert's GCN + pragma MLP weights to use (default 0)")
    ap.add_argument("--list", action="store_true",
                    help="list the kernels in the dataset and exit")
    args = ap.parse_args()

    ddir = find_dataset_dir()
    samples = load_all(ddir)
    if not samples:
        raise SystemExit(f"no data_*.pt in {ddir}")

    if args.list:
        seen = {}
        for _, d in samples:
            k = str(getattr(d, "kernel", "?"))
            k = k[0] if isinstance(k, list) else k
            seen.setdefault(k, [0, d.x.shape[0], d.edge_index.shape[1]])
            seen[k][0] += 1
        print(f"{'kernel':28s} {'cand':>6s} {'N':>6s} {'E':>6s}   fits?")
        for k in sorted(seen):
            c, n, e = seen[k]
            fit = "yes" if (n <= N_MAX and e <= E_MAX) else "NO"
            print(f"{k:28s} {c:6d} {n:6d} {e:6d}   {fit}")
        return

    # ---- pick the graph -------------------------------------------------
    if args.graph:
        hits = [(f, d) for f, d in samples
                if args.graph in str(getattr(d, "kernel", ""))]
        if not hits:
            raise SystemExit(f"no kernel matching '{args.graph}'. Try --list.")
        gfile, g = hits[0]
        ncand = len(hits)
    else:
        gfile, g = max(samples, key=lambda t: t[1].x.shape[0])
        ncand = sum(1 for _, d in samples if d.x.shape[0] == g.x.shape[0])

    N, E = g.x.shape[0], g.edge_index.shape[1]
    print(f"dataset    : {ddir}")
    print(f"graph      : {os.path.basename(gfile)}   ({len(samples)} samples total)")
    print(f"kernel     : {getattr(g, 'kernel', '?')}")
    print(f"candidates : {ncand}")
    print(f"N, E       : {N}, {E}")
    if N > N_MAX or E > E_MAX:
        raise SystemExit(f"graph exceeds the kernel's fixed {N_MAX}/{E_MAX} bounds")

    # ---- weights --------------------------------------------------------
    ck = find_checkpoint()
    sd = torch.load(ck, map_location="cpu")
    print(f"checkpoint : {ck}")
    print(f"expert     : {args.expert}")

    names = ["conv_first"] + [f"conv_layers.{i}" for i in range(N_LAYERS - 1)]
    weights, biases = [], []
    for nm in names:
        pre = f"experts.{args.expert}.{nm}"
        weights.append(sd[f"{pre}.conv.lin.weight"].numpy().astype(np.float32))
        biases.append(sd[f"{pre}.conv.bias"].numpy().astype(np.float32))
    print("weights    : " + ", ".join(str(w.shape) for w in weights))

    os.makedirs(OUT, exist_ok=True)

    # ---- inputs ---------------------------------------------------------
    x = g.x.float().numpy()
    assert x.shape[1] == IN_DIM_REAL, f"expected {IN_DIM_REAL} features, got {x.shape[1]}"

    x_pad = np.zeros((N_MAX, IN_DIM_PAD), dtype=np.float32)
    x_pad[:N, :IN_DIM_REAL] = x

    el = np.zeros((E_MAX, 2), dtype=np.int32)
    el[:E, 0] = g.edge_index[0].numpy()      # source
    el[:E, 1] = g.edge_index[1].numpy()      # dest

    # graph_info = [N, E, expert]; the testbench passes expert to the kernel so
    # the pragma MLP uses the same expert as the GCN weights written below
    np.array([N, E, args.expert], dtype=np.int32).tofile(os.path.join(OUT, "graph_info.bin"))
    x_pad.tofile(os.path.join(OUT, "node_features.bin"))
    el.tofile(os.path.join(OUT, "edge_list.bin"))

    # pragma inputs: X_pragma_per_node [N][4], X_pragmascopenids [N] (0/1)
    xp = np.zeros((N_MAX, 4), dtype=np.float32)
    xp[:N] = g.X_pragma_per_node.float().numpy()
    scope_f = g.X_pragmascopenids.float().numpy().reshape(-1)
    assert set(np.unique(scope_f)) <= {0.0, 1.0}, "X_pragmascopenids is not 0/1"
    scope = np.zeros(N_MAX, dtype=np.int32)
    scope[:N] = scope_f.astype(np.int32)
    xp.tofile(os.path.join(OUT, "pragma_per_node.bin"))
    scope.tofile(os.path.join(OUT, "pragma_scope.bin"))
    print(f"scope nodes: {int(scope.sum())} of {N}")

    # the kernel's pragma weights are compiled in from pragma_weights.h --
    # make sure that header came from this same checkpoint
    hdr = open(os.path.join(HERE, "pragma_weights.h")).read()
    key = f"e{args.expert}_merge_l1_b"
    body = hdr.split(f"static const float {key}[64] = {{", 1)[1].split("};", 1)[0]
    hv = np.array([float(t.strip().rstrip("f")) for t in body.split(",") if t.strip()],
                  dtype=np.float32)
    cv = sd[f"experts.{args.expert}.MLPs_per_pragma.merge.layers.1.bias"].numpy().astype(np.float32)
    if not np.array_equal(hv, cv):
        raise SystemExit("pragma_weights.h does not match this checkpoint -- "
                         "run gen_pragma_weights.py first")

    # ---- weights, every layer zero-padded to [64][96] --------------------
    # The top declares all six weight ports as [64][96]; layers 1-5 only read
    # columns 0..63, so the pad is inert there.
    weights_pad = []
    for i in range(N_LAYERS):
        wp = np.zeros((D, IN_DIM_PAD), dtype=np.float32)
        wp[:, :weights[i].shape[1]] = weights[i]
        wp.tofile(os.path.join(OUT, f"gnn_convs_{i}_conv_lin_weight.bin"))
        biases[i].tofile(os.path.join(OUT, f"gnn_convs_{i}_conv_bias.bin"))
        weights_pad.append(wp)

    # ---- golden ---------------------------------------------------------
    print("\ncomputing golden (6 layers + JK-max) ...")
    jk, per_layer, indeg = build_reference(x_pad[:N], g.edge_index, weights_pad, biases)
    out = np.zeros((N_MAX, D), dtype=np.float32)
    out[:N] = jk
    out.tofile(os.path.join(OUT, "golden_jk.bin"))
    print(f"  JK range  [{jk.min():.6f}, {jk.max():.6f}]   mean {jk.mean():.6f}")
    for i, h in enumerate(per_layer):
        print(f"  layer {i}   range [{h.min():.6f}, {h.max():.6f}]   mean {h.mean():.6f}")

    print("\ncomputing golden (pragma MLP, parallel_and_merge) ...")
    pm = pragma_reference(jk, xp[:N], scope[:N], sd, args.expert)
    out = np.zeros((N_MAX, D), dtype=np.float32)
    out[:N] = pm
    out.tofile(os.path.join(OUT, "golden_pragma.bin"))
    changed = np.abs(pm - jk).max(axis=1) > 0
    print(f"  rows changed by pragma MLP: {int(changed.sum())}  "
          f"(scope nodes: {int(scope[:N].sum())})")

    # how much JK actually draws from each layer -- shows whether the layer-0
    # and layer-5 fixes matter numerically on this graph
    stack = np.stack(per_layer)                      # (6, N, D)
    winner = stack.argmax(axis=0)
    print("\n  JK argmax share per layer:")
    for i in range(N_LAYERS):
        pct = 100.0 * (winner == i).sum() / winner.size
        print(f"    layer {i}: {pct:5.1f}%")

    # ---- degree stats ---------------------------------------------------
    over = int((indeg > 4).sum())
    print(f"\nin-degree  : mean {indeg.mean():.2f}   max {int(indeg.max())}")
    print(f"             {over} of {N} nodes ({100.0 * over / N:.1f}%) have >4 neighbours")
    print(f"             (these are the nodes the old `int neighbors[4]` overflowed on)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
