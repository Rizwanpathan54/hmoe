# GCN + JumpingKnowledge + pragma MLPs — integration

**Scope:** C simulation of the first two stages of one HierarchicalMoE expert:

```
6-layer GCN → JumpingKnowledge(max) → pragma MLPs (tile, pipeline, parallel → merge)
```

The output is the author's `return_middle == 6.3`, which is after the pragma MLPs
and before gnn7 (`model.py:571`). gnn7, attention pooling and the output MLPs
are not part of this. Synthesis has not been attempted; the open csynth issues
from `gcn_conv_jk/CHANGES.md` still apply.

**Result:** C simulation passes for all three experts on `fdtd-2d-large`
(N=555, E=1965, 8 pragma-scope nodes), and for expert 2 on `syr2k` (N=293, E=1024):

| run | stage 1 (JK) MAE | stage 2 (pragma) MAE, scope nodes | non-scope pass-through |
|---|---|---|---|
| fdtd-2d-large, expert 0 | 1.84e-08 | 1.55e-06 | 547/547 exact |
| fdtd-2d-large, expert 1 | 2.23e-08 | 9.91e-07 | 547/547 exact |
| fdtd-2d-large, expert 2 | 2.27e-08 | 1.62e-06 | 547/547 exact |
| syr2k, expert 2 | 1.28e-08 | 1.61e-06 | 289/289 exact |

Stage 1 for expert 0 is identical to `gcn_conv_jk`'s result (1.844931e-08), so
the backbone is unchanged. Logs are in `csim_*.log`.

The two source folders, `~/gcn_conv_jk` and `~/src`, were **not modified**.

---

## Where each file came from

| file | source | status |
|---|---|---|
| `model.cpp`, `model.h` | `gcn_conv_jk` | **modified**: new top, pragma stage (see below) |
| `model_tb.cpp` | `gcn_conv_jk` | **rewritten**: checks both stages |
| `dump_tb_data.py` | `gcn_conv_jk` | **extended**: pragma inputs and pragma golden |
| `hls_config.cfg` | `gcn_conv_jk` | **modified**: new top, both `.cpp` files, new paths |
| `gnn_builder_lib.h` | `gcn_conv_jk` | unchanged |
| `pragma_mlp.cpp`, `pragma_mlp.h` | `~/src` | **modified**: body split out (see below), maths unchanged |
| `pragma_weights.h` | — | **regenerated** by `gen_pragma_weights.py` |
| `pragma_weights_from_teammate.h.orig` | `~/src/pragma_weights.h` | byte-identical copy, kept for reference |
| `gen_pragma_weights.py` | new | builds `pragma_weights.h` from the checkpoint |
| `validate_pragma_reference.py` | new | checks the pragma golden against the real model |

Every code change is marked `INTEGRATION:` in place:

```bash
grep -n "INTEGRATION" *.cpp *.h
```

---

## The changes

### 1. Pragma weights regenerated from the GCN's checkpoint

The teammate's `pragma_weights.h` says it came from `500_test_model_state_dict.pth`.
It matches none of the four checkpoints on this machine: three `500_test` runs
(GAT, GCN, 07-01) and `gnn-gcn/1000_val`. It also matches none of the 26 `.pth`
files under `~`. The GCN and pragma MLPs are trained together inside one
`Net`, so their weights must come from the same checkpoint.

`gen_pragma_weights.py` writes the pragma MLP weights from
`gnn-gcn/run1/1000_val_model_state_dict.pth`, the same file `dump_tb_data.py`
takes the GCN weights from. It checks two things on every run:

- every array reads back **bit-exact** from the checkpoint
- array **names, order and sizes are identical** to the teammate's header, so
  `pragma_mlp.cpp` compiles unchanged against it

`dump_tb_data.py` also refuses to run if `pragma_weights.h` doesn't match the
checkpoint.

### 2. `pragma_mlp()` split into a callable node function — `pragma_mlp.cpp`

The teammate's `pragma_mlp()` is a top function with `m_axi`/`s_axilite`
INTERFACE pragmas. Its body is now `pragma_mlp_node()`, with no interface
pragmas and `INLINE off`. `pragma_mlp()` is kept as a wrapper with the
original signature and pragmas, so the file still works standalone.
`run_mlp`, `run_expert` and the scope/expert logic are untouched.

### 3. New top `HMoE_GCN_JK_PragmaMLP_top` — `model.cpp`, `model.h`

- The first 18 ports are the same as `HMoE_Backbone_GCN_top`, in the same order.
- `final_node_embeddings` now holds the **pragma MLP output**.
- New ports are appended at the end:

| port | type | from |
|---|---|---|
| `pragma_per_node_input` | `F_TYPE [555][4]` | `data.X_pragma_per_node` |
| `pragma_scope_input` | `int [555]` | `data.X_pragmascopenids` (0/1) |
| `expert` | `int` | 0..2. **Must match the GCN weights passed in.** |
| `jk_node_embeddings` | `F_TYPE [555][64]` | GCN+JK output, exported so stage 1 can be checked |

The new `compute_pragma_mlp()` runs after `compute_gnn_head()`. It calls
`pragma_mlp_node()` for each node on `node_emb_out[i]`. Values go through
`float` buffers because `pragma_mlp_node` is `float`-only, which stays correct
if `F_TYPE` becomes fixed point.

---

## What the goldens are

- **`golden_jk.bin`** is unchanged from `gcn_conv_jk`: plain GCN with
  self-loops, ReLU, and JK max, using the kernel's semantics.
- **`golden_pragma.bin`** is `pragma_reference()` in `dump_tb_data.py`. It
  follows the author's `Net.apply_pragam_as_MLP` (`model.py:425-463`,
  `pragma_order='parallel_and_merge'`) line for line, masks included, and is
  applied to `golden_jk`.

`validate_pragma_reference.py` checks `pragma_reference()` against the
**author's real model**. It loads `HierarchicalMoE` with the same checkpoint,
captures each expert's real JK output with a hook, and compares against the
model's own `return_middle=6.3` output:

```
4 graphs × 3 experts:  worst relative error 4.1e-07   PASS
```

The leftover error is float64 vs float32 rounding. The same computation in
float32 gives **0.0** difference, and non-scope rows are exact.

---

## Running it

```bash
cd ~/gcn_jk_mlp

# needs the venv that can import the author's repo (klepto, torch_geometric)
PY=~/capstone1/venv/bin/python

$PY gen_pragma_weights.py                 # only if the checkpoint changes
$PY validate_pragma_reference.py          # optional: golden vs real model
$PY dump_tb_data.py --expert 0            # or 1, 2; --graph syr2k

export PATH=$HOME/xilinx/2025.1/Vitis/bin:$PATH
vitis-run --mode hls --csim --config hls_config.cfg \
          --work_dir ~/fpga_accel/vitis_workspace/gcn_jk_mlp
```

`tb_data/` is currently set to expert 0, `fdtd-2d-large`.

---

## Known limits (deliberately left as they are)

1. **The backbone is not the author's real backbone.** The kernel uses ReLU
   and no `lin_edge`. The model uses ELU and `lin_edge` edge weights. Both
   kernel stages are correct against the kernel-semantics golden. End-to-end
   agreement with the real model needs the backbone fixed first. This was
   already the case in `gcn_conv_jk`.
2. **All three experts' pragma MLPs are built into the hardware.** The
   teammate's `run_expert<0/1/2>` creates 12 MLP instances and about 450 KB of
   weight ROM. The GCN, by contrast, takes one expert's weights through ports.
   One approach should be chosen before csynth.
3. **One expert per call.** All three branches mean calling the top three
   times, each with that expert's GCN weights and the matching `expert`.
4. **The csynth issues from `gcn_conv_jk/CHANGES.md` are still open**:
   `conv_next_in` BRAM/URAM, `NODE_LOOP` II=1, and URAM capacity.
