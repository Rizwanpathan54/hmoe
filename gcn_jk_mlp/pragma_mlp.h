#ifndef PRAGMA_MLP_H
#define PRAGMA_MLP_H

// One node through the pragma MLPs (tile, pipeline, parallel -> merge) of the
// chosen expert. Non-scope nodes pass through unchanged.
void pragma_mlp_node(
    const float embedding[64],
    const float pragma_features[4],
    int expert,
    int scope,
    float output[64]
);

// Standalone top, same signature as the teammate's original.
void pragma_mlp(
    const float embedding[64],
    const float pragma_features[4],
    int expert,
    int scope,
    float output[64]
);

#endif
