// ============================================================================
//  Testbench -- GCN + JumpingKnowledge backbone + pragma MLPs
//
//  Loads tb_data/ (produced by dump_tb_data.py) and checks both stages of
//  HMoE_GCN_JK_PragmaMLP_top against PyTorch-derived goldens:
//
//    stage 1  jk_node_embeddings     vs golden_jk.bin
//    stage 2  final_node_embeddings  vs golden_pragma.bin
//
//  WHAT THE GOLDENS ARE
//  --------------------
//  golden_jk: element-wise MAX over the six ReLU'd layer outputs of a 6-layer
//  GCN chain with self-loops, using the trained conv weights:
//
//      neighbour j -> i :  1 / sqrt((d_i + 1)(d_j + 1))
//      self-loop        :  1 / (d_i + 1)
//
//  It deliberately excludes lin_edge and ELU -- the kernel has neither.
//
//  golden_pragma: the author's Net.apply_pragam_as_MLP (parallel_and_merge)
//  applied to golden_jk, with the same expert's pragma MLP weights. That
//  function is checked against the real model by validate_pragma_reference.py.
//
//  Only the first num_of_nodes rows are compared; rows beyond that are padding
//  the kernel never writes.
// ============================================================================
#include "model.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define N_MAX   555
#define E_MAX  1965
#define IN_DIM   96
#define EMB      64
#define P_DIM     4

static const char *DD = "tb_data/";

static std::string P(const char *f) { return std::string(DD) + f; }

template <typename T>
static bool rd(const char *fname, T *dst, size_t count)
{
    std::string path = P(fname);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path.c_str()); return false; }
    size_t got = fread(dst, sizeof(T), count, f);
    fclose(f);
    if (got != count) {
        printf("ERROR: %s -- read %zu of %zu elements\n", path.c_str(), got, count);
        return false;
    }
    return true;
}

// ---- kernel I/O (static: ~1.5 MB total, far too big for the stack) --------
static F_TYPE x_in      [N_MAX][IN_DIM];
static int    el_in     [E_MAX][2];
static F_TYPE pragma_in [N_MAX][P_DIM];
static int    scope_in  [N_MAX];
static F_TYPE jk_out    [N_MAX][EMB];
static F_TYPE final_out [N_MAX][EMB];
static F_TYPE golden_jk [N_MAX][EMB];
static F_TYPE golden_pm [N_MAX][EMB];

static W_TYPE w0[EMB][IN_DIM], w1[EMB][IN_DIM], w2[EMB][IN_DIM];
static W_TYPE w3[EMB][IN_DIM], w4[EMB][IN_DIM], w5[EMB][IN_DIM];
static W_TYPE b0[EMB], b1[EMB], b2[EMB], b3[EMB], b4[EMB], b5[EMB];

struct Cmp { double mae, max_abs; int wi, wj; bool nan; };

static Cmp compare(F_TYPE got[N_MAX][EMB], F_TYPE exp_[N_MAX][EMB], int N,
                   const int *row_mask, int want)
{
    Cmp c = {0.0, 0.0, 0, 0, false};
    double sum = 0.0;
    long cnt = 0;
    for (int i = 0; i < N; i++) {
        if (row_mask && row_mask[i] != want) continue;
        for (int j = 0; j < EMB; j++) {
            double g = (double)got[i][j], e = (double)exp_[i][j];
            if (std::isnan(g) || std::isinf(g)) c.nan = true;
            double d = std::fabs(g - e);
            sum += d;
            cnt++;
            if (d > c.max_abs) { c.max_abs = d; c.wi = i; c.wj = j; }
        }
    }
    c.mae = cnt ? sum / cnt : 0.0;
    return c;
}

int main()
{
    int info[3];
    if (!rd("graph_info.bin", info, 3)) return 1;
    const int N = info[0], E = info[1], EXPERT = info[2];
    printf("graph       : N = %d, E = %d, expert = %d\n", N, E, EXPERT);

    if (N > N_MAX || E > E_MAX) {
        printf("ERROR: graph exceeds the kernel's fixed %d/%d bounds\n", N_MAX, E_MAX);
        return 1;
    }

    bool ok = true;
    ok &= rd("node_features.bin",   &x_in[0][0],      (size_t)N_MAX * IN_DIM);
    ok &= rd("edge_list.bin",       &el_in[0][0],     (size_t)E_MAX * 2);
    ok &= rd("pragma_per_node.bin", &pragma_in[0][0], (size_t)N_MAX * P_DIM);
    ok &= rd("pragma_scope.bin",    scope_in,         (size_t)N_MAX);
    ok &= rd("golden_jk.bin",       &golden_jk[0][0], (size_t)N_MAX * EMB);
    ok &= rd("golden_pragma.bin",   &golden_pm[0][0], (size_t)N_MAX * EMB);

    ok &= rd("gnn_convs_0_conv_lin_weight.bin", &w0[0][0], (size_t)EMB * IN_DIM);
    ok &= rd("gnn_convs_1_conv_lin_weight.bin", &w1[0][0], (size_t)EMB * IN_DIM);
    ok &= rd("gnn_convs_2_conv_lin_weight.bin", &w2[0][0], (size_t)EMB * IN_DIM);
    ok &= rd("gnn_convs_3_conv_lin_weight.bin", &w3[0][0], (size_t)EMB * IN_DIM);
    ok &= rd("gnn_convs_4_conv_lin_weight.bin", &w4[0][0], (size_t)EMB * IN_DIM);
    ok &= rd("gnn_convs_5_conv_lin_weight.bin", &w5[0][0], (size_t)EMB * IN_DIM);

    ok &= rd("gnn_convs_0_conv_bias.bin", b0, EMB);
    ok &= rd("gnn_convs_1_conv_bias.bin", b1, EMB);
    ok &= rd("gnn_convs_2_conv_bias.bin", b2, EMB);
    ok &= rd("gnn_convs_3_conv_bias.bin", b3, EMB);
    ok &= rd("gnn_convs_4_conv_bias.bin", b4, EMB);
    ok &= rd("gnn_convs_5_conv_bias.bin", b5, EMB);
    if (!ok) return 1;

    int n_scope = 0;
    for (int i = 0; i < N; i++) n_scope += scope_in[i];
    printf("scope nodes : %d\n", n_scope);

    for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < EMB; j++)
            jk_out[i][j] = final_out[i][j] = F_TYPE(0.0);

    printf("running kernel ...\n");
    HMoE_GCN_JK_PragmaMLP_top(
        x_in, el_in, final_out,
        N, E,
        1,                       // copy_parameters_flag
        b0, w0, b1, w1, b2, w2,
        b3, w3, b4, w4, b5, w5,
        pragma_in, scope_in, EXPERT,
        jk_out);

    const double TOL = 1e-4;
    bool pass = true;

    // ---- stage 1: GCN + JK ------------------------------------------------
    Cmp s1 = compare(jk_out, golden_jk, N, nullptr, 0);
    printf("\n--- stage 1  JK embeddings: kernel vs PyTorch -----------------\n");
    printf("compared    : %d nodes x %d dims = %d values\n", N, EMB, N * EMB);
    printf("MAE         : %.6e\n", s1.mae);
    printf("max |err|   : %.6e   at node %d, dim %d\n", s1.max_abs, s1.wi, s1.wj);
    pass &= !s1.nan && s1.mae < TOL;

    // ---- stage 2: pragma MLP ----------------------------------------------
    Cmp s2  = compare(final_out, golden_pm, N, nullptr, 0);
    Cmp s2s = compare(final_out, golden_pm, N, scope_in, 1);
    printf("\n--- stage 2  pragma MLP output: kernel vs PyTorch --------------\n");
    printf("all nodes   : MAE %.6e   max |err| %.6e at node %d, dim %d\n",
           s2.mae, s2.max_abs, s2.wi, s2.wj);
    printf("scope nodes : MAE %.6e   max |err| %.6e at node %d, dim %d\n",
           s2s.mae, s2s.max_abs, s2s.wi, s2s.wj);
    if (n_scope > 0)
        printf("              kernel %.6f   golden %.6f\n",
               (double)final_out[s2s.wi][s2s.wj], (double)golden_pm[s2s.wi][s2s.wj]);
    pass &= !s2.nan && s2.mae < TOL && s2s.mae < TOL;

    // non-scope nodes must pass the JK embedding through bit-exactly
    int passthru_bad = 0;
    for (int i = 0; i < N; i++)
        if (!scope_in[i])
            for (int j = 0; j < EMB; j++)
                if (final_out[i][j] != jk_out[i][j]) { passthru_bad++; break; }
    printf("non-scope   : %d of %d rows differ from the JK output (expect 0)\n",
           passthru_bad, N - n_scope);
    pass &= passthru_bad == 0;

    if (pass) {
        printf("\nRESULT: PASS  (stage 1 and stage 2 MAE < %.0e)\n", TOL);
        return 0;
    }
    printf("\nRESULT: FAIL\n");
    return 1;
}
