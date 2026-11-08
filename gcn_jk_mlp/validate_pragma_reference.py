#!/usr/bin/env python3
"""
Checks dump_tb_data.pragma_reference() against the author's real model.

Loads the trained HierarchicalMoE (same checkpoint, same loading recipe as
~/fpgahls_accel/ref/dump_reference.py), captures each expert's REAL
JumpingKnowledge output with a forward hook, runs the author's forward up to
return_middle=6.3 (after the pragma MLPs, before gnn7), and compares that with
pragma_reference() applied to the captured JK output.

This isolates the pragma stage: the backbone differences between the kernel
and the real model (ReLU vs ELU, no lin_edge) don't enter, because both sides
start from the same real JK tensor.

Usage
    cd ~/gcn_jk_mlp
    source ~/fpga_accel/venv/bin/activate
    python validate_pragma_reference.py                 # graphs 0, 100, 500
    python validate_pragma_reference.py --graphs 0 7
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dump_tb_data import pragma_reference   # noqa: E402

REPO = "/home/rizwanpathan/hierarchical_moe/HierarchicalMoE"
SRC = os.path.join(REPO, "src")


def load_model():
    sys.argv = ["validate_pragma_reference", "--gnn_type", "gcn"]
    sys.path.insert(0, SRC)
    cwd = os.getcwd()
    os.chdir(SRC)
    try:
        from config import FLAGS
        FLAGS.device = "cpu"
        from model import HierarchicalMoE

        assert FLAGS.pragma_order == "parallel_and_merge", FLAGS.pragma_order
        assert FLAGS.activation == "elu", FLAGS.activation

        ck = sorted(glob.glob("logs/**/*gnn-gcn/run1/1000_val_model_state_dict.pth",
                              recursive=True))[0]
        files = sorted(glob.glob(os.path.join(REPO, "save/programl/*/finetune_dataset/data_*.pt")))
        d0 = torch.load(files[0], map_location="cpu", weights_only=False)
        model = HierarchicalMoE(d0.x.shape[1], edge_dim=d0.edge_attr.shape[1],
                                init_pragma_dict=None, task="regression").to("cpu")
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, sd, files
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, nargs="+", default=[0, 100, 500])
    args = ap.parse_args()

    from torch_geometric.loader import DataLoader
    model, sd, files = load_model()

    worst = 0.0
    for gi in args.graphs:
        data = torch.load(files[gi], map_location="cpu", weights_only=False)
        batch = next(iter(DataLoader([data], batch_size=1)))

        cap = {}
        hooks = [model.experts[i].jkn.register_forward_hook(
                     lambda m, inp, o, i=i: cap.__setitem__(i, o.detach().clone()))
                 for i in range(3)]
        with torch.no_grad():
            real = [model.experts[i](batch, 6.3, "", True) for i in range(3)]
        for h in hooks:
            h.remove()

        xp = batch.X_pragma_per_node.float().numpy()
        scope = batch.X_pragmascopenids.float().numpy().reshape(-1)
        print(f"graph {gi:4d}  {os.path.basename(files[gi])}  N={xp.shape[0]}  "
              f"scope nodes={int(scope.sum())}")
        for i in range(3):
            ref = pragma_reference(cap[i].numpy(), xp, scope, sd, i)
            r = real[i].numpy()
            err = np.abs(ref - r).max()
            # error relative to the output's scale: scope rows reach |~60|, and
            # the reference is float64 while the model is float32
            rel = err / np.abs(r).max()
            worst = max(worst, rel)
            print(f"   expert {i}: max |err| = {err:.3e}   relative to max|out| = {rel:.3e}")

    tol = 1e-6
    print(f"\nRESULT: {'PASS' if worst < tol else 'FAIL'}  (worst relative {worst:.3e}, tol {tol:.0e})")
    sys.exit(0 if worst < tol else 1)


if __name__ == "__main__":
    main()
