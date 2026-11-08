#include "pragma_mlp.h"
#include "pragma_weights.h"
#include <math.h>

template<int IN, int HIDDEN, int OUT>
static void run_mlp(
    const float input[IN],
    const float w1[HIDDEN * IN],
    const float b1[HIDDEN],
    const float w2[OUT * HIDDEN],
    const float b2[OUT],
    float output[OUT]
) {
#pragma HLS INLINE off
    float hidden[HIDDEN];

    // First Linear layer followed by ELU.
    for (int i = 0; i < HIDDEN; ++i) {
#pragma HLS PIPELINE II=1
        float sum = b1[i];
        for (int j = 0; j < IN; ++j) {
            sum += w1[i * IN + j] * input[j];
        }
        hidden[i] = (sum >= 0.0f) ? sum : expf(sum) - 1.0f;
    }

    // Final Linear layer: no activation.
    for (int i = 0; i < OUT; ++i) {
#pragma HLS PIPELINE II=1
        float sum = b2[i];
        for (int j = 0; j < HIDDEN; ++j) {
            sum += w2[i * HIDDEN + j] * hidden[j];
        }
        output[i] = sum;
    }
}

template<int EXPERT>
static void run_expert(
    const float embedding[64],
    const float pragma_features[4],
    float output[64]
) {
    float tile_input[65];
    float pipeline_input[65];
    float parallel_input[66];

    for (int i = 0; i < 64; ++i) {
        tile_input[i] = embedding[i];
        pipeline_input[i] = embedding[i];
        parallel_input[i] = embedding[i];
    }
    tile_input[64] = pragma_features[0];
    pipeline_input[64] = pragma_features[1];
    parallel_input[64] = pragma_features[2];
    parallel_input[65] = pragma_features[3];

    float tile_out[64];
    float pipeline_out[64];
    float parallel_out[64];

    run_mlp<65, 32, 64>(
        tile_input,
        EXPERT == 0 ? e0_tile_l0_w : EXPERT == 1 ? e1_tile_l0_w : e2_tile_l0_w,
        EXPERT == 0 ? e0_tile_l0_b : EXPERT == 1 ? e1_tile_l0_b : e2_tile_l0_b,
        EXPERT == 0 ? e0_tile_l1_w : EXPERT == 1 ? e1_tile_l1_w : e2_tile_l1_w,
        EXPERT == 0 ? e0_tile_l1_b : EXPERT == 1 ? e1_tile_l1_b : e2_tile_l1_b,
        tile_out
    );

    run_mlp<65, 32, 64>(
        pipeline_input,
        EXPERT == 0 ? e0_pipeline_l0_w : EXPERT == 1 ? e1_pipeline_l0_w : e2_pipeline_l0_w,
        EXPERT == 0 ? e0_pipeline_l0_b : EXPERT == 1 ? e1_pipeline_l0_b : e2_pipeline_l0_b,
        EXPERT == 0 ? e0_pipeline_l1_w : EXPERT == 1 ? e1_pipeline_l1_w : e2_pipeline_l1_w,
        EXPERT == 0 ? e0_pipeline_l1_b : EXPERT == 1 ? e1_pipeline_l1_b : e2_pipeline_l1_b,
        pipeline_out
    );

    run_mlp<66, 33, 64>(
        parallel_input,
        EXPERT == 0 ? e0_parallel_l0_w : EXPERT == 1 ? e1_parallel_l0_w : e2_parallel_l0_w,
        EXPERT == 0 ? e0_parallel_l0_b : EXPERT == 1 ? e1_parallel_l0_b : e2_parallel_l0_b,
        EXPERT == 0 ? e0_parallel_l1_w : EXPERT == 1 ? e1_parallel_l1_w : e2_parallel_l1_w,
        EXPERT == 0 ? e0_parallel_l1_b : EXPERT == 1 ? e1_parallel_l1_b : e2_parallel_l1_b,
        parallel_out
    );

    float merge_input[192];
    for (int i = 0; i < 64; ++i) {
        merge_input[i]       = tile_out[i];
        merge_input[64 + i]  = pipeline_out[i];
        merge_input[128 + i] = parallel_out[i];
    }

    run_mlp<192, 96, 64>(
        merge_input,
        EXPERT == 0 ? e0_merge_l0_w : EXPERT == 1 ? e1_merge_l0_w : e2_merge_l0_w,
        EXPERT == 0 ? e0_merge_l0_b : EXPERT == 1 ? e1_merge_l0_b : e2_merge_l0_b,
        EXPERT == 0 ? e0_merge_l1_w : EXPERT == 1 ? e1_merge_l1_w : e2_merge_l1_w,
        EXPERT == 0 ? e0_merge_l1_b : EXPERT == 1 ? e1_merge_l1_b : e2_merge_l1_b,
        output
    );
}

// INTEGRATION: the body that used to live in pragma_mlp() moved here, with no
// INTERFACE pragmas, so HMoE_GCN_JK_PragmaMLP_top can call it once per node.
// The maths is unchanged.
void pragma_mlp_node(
    const float embedding[64],
    const float pragma_features[4],
    int expert,
    int scope,
    float output[64]
) {
#pragma HLS INLINE off
    if (!scope) {
        for (int i = 0; i < 64; ++i) {
            output[i] = embedding[i];
        }
        return;
    }

    if (expert == 0) {
        run_expert<0>(embedding, pragma_features, output);
    } else if (expert == 1) {
        run_expert<1>(embedding, pragma_features, output);
    } else if (expert == 2) {
        run_expert<2>(embedding, pragma_features, output);
    } else {
        for (int i = 0; i < 64; ++i) output[i] = 0.0f;
    }
}

// Standalone top, kept so this file still builds on its own as before.
void pragma_mlp(
    const float embedding[64],
    const float pragma_features[4],
    int expert,
    int scope,
    float output[64]
) {
#pragma HLS INTERFACE s_axilite port=expert bundle=control
#pragma HLS INTERFACE s_axilite port=scope bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi port=embedding offset=slave bundle=gmem depth=64
#pragma HLS INTERFACE m_axi port=pragma_features offset=slave bundle=gmem depth=4
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem depth=64

    pragma_mlp_node(embedding, pragma_features, expert, scope, output);
}
