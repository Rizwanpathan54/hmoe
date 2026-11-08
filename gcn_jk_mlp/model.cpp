#include "model.h"

// --- Global Weights (Bound to URAM to save BRAM) ---

W_TYPE gnn_convs_0_conv_bias_fixed[64];
W_TYPE gnn_convs_0_conv_lin_weight_fixed[64][96];
W_TYPE gnn_convs_1_conv_bias_fixed[64];
W_TYPE gnn_convs_1_conv_lin_weight_fixed[64][96];
W_TYPE gnn_convs_2_conv_bias_fixed[64];
W_TYPE gnn_convs_2_conv_lin_weight_fixed[64][96];
W_TYPE gnn_convs_3_conv_bias_fixed[64];
W_TYPE gnn_convs_3_conv_lin_weight_fixed[64][96];
W_TYPE gnn_convs_4_conv_bias_fixed[64];
W_TYPE gnn_convs_4_conv_lin_weight_fixed[64][96];
W_TYPE gnn_convs_5_conv_bias_fixed[64];
W_TYPE gnn_convs_5_conv_lin_weight_fixed[64][96];

// Input Graph Data
F_TYPE node_feature_table[555][96];
F_TYPE node_feature_table_copy[555][96];
int edge_list[1965][2];

// Neighbor Tables
int neightbor_table_offsets[555];
int neighbor_table[1965];
int in_degree_table[555];
int out_degree_table[1965];

// Final GCN Backbone & JK-Max Output Buffer (Initialized to zero)
F_TYPE node_emb_out[555][64] = {0};

// Helper function to update the running Max Jumping Knowledge buffer sequentially
void update_jk_max(F_TYPE layer_output[555][96], int layer_index, int n_nodes) {
    #pragma HLS INLINE off
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        for (int j = 0; j < 64; j++) {
            //#pragma HLS PIPELINE II=1
            #pragma HLS UNROLL
            F_TYPE val = layer_output[i][j];
            if (layer_index == 0) {
                // Layer 0: Initialize running accumulator directly
                node_emb_out[i][j] = val;
            } else {
                // Layers 1 to 5: Keep element-wise maximum
                if (val > node_emb_out[i][j]) {
                    node_emb_out[i][j] = val;
                }
            }
        }
    }
}

void compute_gnn_head(int n_nodes, int n_edges){
    #pragma HLS INLINE off

    // Temporary working buffers for active layers
    F_TYPE conv_0_post_conv[555][96];
    #pragma HLS ARRAY_PARTITION variable=conv_0_post_conv dim=2
    #pragma HLS bind_storage variable=conv_0_post_conv type=RAM_2P impl=uram

    F_TYPE conv_next_in[555][96];
    #pragma HLS ARRAY_PARTITION variable=conv_next_in dim=2
    #pragma HLS bind_storage variable=conv_next_in type=RAM_2P impl=bram

    F_TYPE conv_active_out[555][96];
    #pragma HLS ARRAY_PARTITION variable=conv_active_out dim=2
    #pragma HLS bind_storage variable=conv_active_out type=RAM_2P impl=uram

    // FIX (b): this whole block was commented out. Layer 1 below consumes
    // conv_next_in and conv_0_post_conv, which nothing else writes -- so
    // without layer 0 the five remaining layers ran on uninitialised stack
    // memory, and JK never got its layer_index==0 initialisation either.
    // --- CONV LAYER 0 (96 -> 64) ---
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        96, n_nodes, n_edges, node_feature_table, node_feature_table_copy, conv_0_post_conv,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_0_conv_lin_weight_fixed, gnn_convs_0_conv_bias_fixed
    );

    // Apply activation, populate next layer input, and update JK-Max (Layer 0)
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            F_TYPE act = activation_relu(conv_0_post_conv[i][j]);
            conv_next_in[i][j] = act;
            conv_0_post_conv[i][j] = act; // Keep residual base if needed
        }
    }
    update_jk_max(conv_next_in, 0, n_nodes);

    // --- CONV LAYER 1 (64 -> 64 with Residual) ---
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        64, n_nodes, n_edges, conv_next_in, conv_0_post_conv, conv_active_out,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_1_conv_lin_weight_fixed, gnn_convs_1_conv_bias_fixed
    );
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            F_TYPE act = activation_relu(conv_active_out[i][j]);
            conv_next_in[i][j] = act;
            conv_0_post_conv[i][j] = act;
        }
    }
    update_jk_max(conv_next_in, 1, n_nodes);

    // --- CONV LAYER 2 (64 -> 64 with Residual) ---
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        64, n_nodes, n_edges, conv_next_in, conv_0_post_conv, conv_active_out,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_2_conv_lin_weight_fixed, gnn_convs_2_conv_bias_fixed
    );
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            F_TYPE act = activation_relu(conv_active_out[i][j]);
            conv_next_in[i][j] = act;
            conv_0_post_conv[i][j] = act;
        }
    }
    update_jk_max(conv_next_in, 2, n_nodes);

    // --- CONV LAYER 3 (64 -> 64 with Residual) ---
    // FIX (e): was W_TYPE. Identical in this build (both typedef to float),
    // but it would break a fixed-point port where F_TYPE != W_TYPE.
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        64, n_nodes, n_edges, conv_next_in, conv_0_post_conv, conv_active_out,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_3_conv_lin_weight_fixed, gnn_convs_3_conv_bias_fixed
    );
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            F_TYPE act = activation_relu(conv_active_out[i][j]);
            conv_next_in[i][j] = act;
            conv_0_post_conv[i][j] = act;
        }
    }
    update_jk_max(conv_next_in, 3, n_nodes);

    // --- CONV LAYER 4 (64 -> 64 with Residual) ---
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        64, n_nodes, n_edges, conv_next_in, conv_0_post_conv, conv_active_out,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_4_conv_lin_weight_fixed, gnn_convs_4_conv_bias_fixed
    );
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            F_TYPE act = activation_relu(conv_active_out[i][j]);
            conv_next_in[i][j] = act;
            conv_0_post_conv[i][j] = act;
        }
    }
    update_jk_max(conv_next_in, 4, n_nodes);

    // --- CONV LAYER 5 (Final Layer) ---
    gcn_conv<555, 1965, 96, 64, F_TYPE, 555, 1965, 4, 16, 16>(
        64, n_nodes, n_edges, conv_next_in, conv_0_post_conv, conv_active_out,
        edge_list, neightbor_table_offsets, neighbor_table,
        in_degree_table, out_degree_table,
        gnn_convs_5_conv_lin_weight_fixed, gnn_convs_5_conv_bias_fixed
    );
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL
            conv_active_out[i][j] = activation_relu(conv_active_out[i][j]);
        }
    }
    // FIX (d): was `update_jk_max(conv_next_in, 5, n_nodes)`. Layer 5's ReLU
    // (just above) writes conv_active_out in place; conv_next_in still holds
    // layer 4. As written, JK collected layer 4 twice and never saw layer 5.
    update_jk_max(conv_active_out, 5, n_nodes);
}

// Load Parameters Function
void load_parameters(
    F_TYPE gnn_convs_0_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_0_conv_lin_weight_fixed_in[64][96],
    F_TYPE gnn_convs_1_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_1_conv_lin_weight_fixed_in[64][96],
    F_TYPE gnn_convs_2_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_2_conv_lin_weight_fixed_in[64][96],
    F_TYPE gnn_convs_3_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_3_conv_lin_weight_fixed_in[64][96],
    F_TYPE gnn_convs_4_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_4_conv_lin_weight_fixed_in[64][96],
    F_TYPE gnn_convs_5_conv_bias_fixed_in[64],
    F_TYPE gnn_convs_5_conv_lin_weight_fixed_in[64][96]
){
    #pragma HLS INLINE off
    copy_1d<64>(gnn_convs_0_conv_bias_fixed_in, gnn_convs_0_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_0_conv_lin_weight_fixed_in, gnn_convs_0_conv_lin_weight_fixed);
    copy_1d<64>(gnn_convs_1_conv_bias_fixed_in, gnn_convs_1_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_1_conv_lin_weight_fixed_in, gnn_convs_1_conv_lin_weight_fixed);
    copy_1d<64>(gnn_convs_2_conv_bias_fixed_in, gnn_convs_2_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_2_conv_lin_weight_fixed_in, gnn_convs_2_conv_lin_weight_fixed);
    copy_1d<64>(gnn_convs_3_conv_bias_fixed_in, gnn_convs_3_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_3_conv_lin_weight_fixed_in, gnn_convs_3_conv_lin_weight_fixed);
    copy_1d<64>(gnn_convs_4_conv_bias_fixed_in, gnn_convs_4_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_4_conv_lin_weight_fixed_in, gnn_convs_4_conv_lin_weight_fixed);
    copy_1d<64>(gnn_convs_5_conv_bias_fixed_in, gnn_convs_5_conv_bias_fixed);
    copy_2d<64, 96>(gnn_convs_5_conv_lin_weight_fixed_in, gnn_convs_5_conv_lin_weight_fixed);
}

// FIX (c): edge_list_in was [555][2] and the copy was copy_2d<555,2>, so only
// the first 555 of up to 1965 edges ever reached the kernel. Worse, the degree
// tables are built over the REAL num_of_edges, so entries 555..1964 of the
// zero-filled global counted as edges 0 -> 0: node 0 picked up ~1400 phantom
// in-edges. Both dimensions are now MAX_EDGES (1965).
void load_graph_data(
    F_TYPE node_feature_table_in[555][96],
    int edge_list_in[1965][2]
){
    #pragma HLS INLINE off
    copy_2d<555, 96>(node_feature_table_in, node_feature_table);
    copy_2d<555, 96>(node_feature_table_in, node_feature_table_copy);
    copy_2d<1965, 2>(edge_list_in, edge_list);
}

// INTEGRATION: pragma MLP stage, run on the JK output node by node.
// Mirrors Net.apply_pragam_as_MLP with pragma_order='parallel_and_merge'
// (model.py:425-463): scope nodes get merge(tile, pipeline, parallel), all
// other nodes keep their JK embedding. pragma_mlp_node takes float, so the
// values are copied through float buffers to stay correct if F_TYPE becomes
// fixed point.
void compute_pragma_mlp(
    F_TYPE pragma_per_node_in[555][4],
    int pragma_scope_in[555],
    int expert,
    int n_nodes,
    F_TYPE pragma_out[555][64]
){
    #pragma HLS INLINE off
    PRAGMA_NODE_LOOP:
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        float emb[64];
        float feats[4];
        float out[64];
        for (int j = 0; j < 64; j++) emb[j] = (float)node_emb_out[i][j];
        for (int j = 0; j < 4; j++) feats[j] = (float)pragma_per_node_in[i][j];

        pragma_mlp_node(emb, feats, expert, pragma_scope_in[i], out);

        for (int j = 0; j < 64; j++) pragma_out[i][j] = (F_TYPE)out[j];
    }
}

// --- TOP KERNEL ---
void HMoE_GCN_JK_PragmaMLP_top(
    F_TYPE node_feature_table_input[555][96],
    int edge_list_input[1965][2],          // FIX (c): was [555][2]
    F_TYPE final_node_embeddings[555][64],
    int num_of_nodes,
    int num_of_edges,
    int copy_parameters_flag,
    W_TYPE gnn_convs_0_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_0_conv_lin_weight_fixed_in[64][96],
    W_TYPE gnn_convs_1_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_1_conv_lin_weight_fixed_in[64][96],
    W_TYPE gnn_convs_2_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_2_conv_lin_weight_fixed_in[64][96],
    W_TYPE gnn_convs_3_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_3_conv_lin_weight_fixed_in[64][96],
    W_TYPE gnn_convs_4_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_4_conv_lin_weight_fixed_in[64][96],
    W_TYPE gnn_convs_5_conv_bias_fixed_in[64],
    W_TYPE gnn_convs_5_conv_lin_weight_fixed_in[64][96],
    F_TYPE pragma_per_node_input[555][4],
    int pragma_scope_input[555],
    int expert,
    F_TYPE jk_node_embeddings[555][64]
){
    #pragma HLS INTERFACE m_axi depth=2220  port=pragma_per_node_input offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=555   port=pragma_scope_input offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=35520 port=jk_node_embeddings offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=88800 port=node_feature_table_input offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=3930  port=edge_list_input offset=slave bundle=mem  // FIX (c): was 1110 (= 555*2)
    #pragma HLS INTERFACE m_axi depth=35520 port=final_node_embeddings offset=slave bundle=mem

    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_0_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=10240 port=gnn_convs_0_conv_lin_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_1_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096  port=gnn_convs_1_conv_lin_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_2_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096  port=gnn_convs_2_conv_lin_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_3_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096  port=gnn_convs_3_conv_lin_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_4_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096  port=gnn_convs_4_conv_lin_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64   port=gnn_convs_5_conv_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096  port=gnn_convs_5_conv_lin_weight_fixed_in offset=slave bundle=mem

    // Bind persistent buffers
    #pragma HLS ARRAY_PARTITION variable=node_feature_table dim=2
    #pragma HLS BIND_STORAGE variable=node_feature_table type=RAM_2P impl=bram
    #pragma HLS ARRAY_PARTITION variable=node_feature_table_copy dim=2
    #pragma HLS BIND_STORAGE variable=node_feature_table_copy type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=edge_list          type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=neighbor_table     type=RAM_2P impl=uram

    #pragma HLS bind_storage variable=node_emb_out       type=RAM_2P impl=uram
    #pragma HLS ARRAY_PARTITION variable=node_emb_out dim=2
    #pragma HLS ARRAY_PARTITION variable=in_degree_table complete

    // ============================================================================
    // WEIGHT ARRAYS PARTITIONING
    // ============================================================================
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_0_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_0_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_0_conv_lin_weight_fixed cyclic factor=16 dim=2

    #pragma HLS ARRAY_PARTITION variable=gnn_convs_1_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_1_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_1_conv_lin_weight_fixed cyclic factor=16 dim=2

    #pragma HLS ARRAY_PARTITION variable=gnn_convs_2_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_2_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_2_conv_lin_weight_fixed cyclic factor=16 dim=2

    #pragma HLS ARRAY_PARTITION variable=gnn_convs_3_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_3_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_3_conv_lin_weight_fixed cyclic factor=16 dim=2

    #pragma HLS ARRAY_PARTITION variable=gnn_convs_4_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_4_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_4_conv_lin_weight_fixed cyclic factor=16 dim=2

    #pragma HLS ARRAY_PARTITION variable=gnn_convs_5_conv_bias_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_5_conv_lin_weight_fixed cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=gnn_convs_5_conv_lin_weight_fixed cyclic factor=16 dim=2

    int num_of_nodes_top = num_of_nodes;
    int num_of_edges_top = num_of_edges;

    if (copy_parameters_flag) {
        load_parameters(
            gnn_convs_0_conv_bias_fixed_in, gnn_convs_0_conv_lin_weight_fixed_in,
            gnn_convs_1_conv_bias_fixed_in, gnn_convs_1_conv_lin_weight_fixed_in,
            gnn_convs_2_conv_bias_fixed_in, gnn_convs_2_conv_lin_weight_fixed_in,
            gnn_convs_3_conv_bias_fixed_in, gnn_convs_3_conv_lin_weight_fixed_in,
            gnn_convs_4_conv_bias_fixed_in, gnn_convs_4_conv_lin_weight_fixed_in,
            gnn_convs_5_conv_bias_fixed_in, gnn_convs_5_conv_lin_weight_fixed_in
        );
    }

    load_graph_data(node_feature_table_input, edge_list_input);

    compute_degree_tables<555, 1965, 555, 1965>(
        edge_list, in_degree_table, out_degree_table,
        num_of_nodes_top, num_of_edges_top
    );

    compute_neighbor_tables<555, 1965, 555, 1965>(
        edge_list, in_degree_table, out_degree_table,
        neightbor_table_offsets, neighbor_table,
        num_of_nodes_top, num_of_edges_top
    );

    // Run the 6 GCN layers with streaming sequential JK-Max updates
    compute_gnn_head(num_of_nodes_top, num_of_edges_top);

    // Export the JK-Max aggregated node embeddings (stage-1 output, for checking)
    for (int i = 0; i < num_of_nodes_top; i++) {
        #pragma HLS loop_tripcount min=1 max=555
        for (int j = 0; j < 64; j++) {
            #pragma HLS PIPELINE II=1
            jk_node_embeddings[i][j] = node_emb_out[i][j];
        }
    }

    // INTEGRATION: pragma MLPs on the JK output, straight into the final output
    compute_pragma_mlp(pragma_per_node_input, pragma_scope_input, expert,
                       num_of_nodes_top, final_node_embeddings);
}
