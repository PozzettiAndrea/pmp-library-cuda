// SPDX-License-Identifier: MIT
#pragma once

#include "pmp/surface_mesh.h"

#include <string>

namespace pmp {

enum class RemeshStrategy { CPU = 0, CUDA = 1 };

struct RemeshingCudaParams
{
    // Per-operation strategy
    RemeshStrategy split_strategy = RemeshStrategy::CPU;
    RemeshStrategy collapse_strategy = RemeshStrategy::CPU;
    RemeshStrategy flip_strategy = RemeshStrategy::CPU;
    RemeshStrategy smooth_strategy = RemeshStrategy::CPU;

    // Iteration counts
    unsigned int outer_iterations = 10;
    unsigned int split_passes = 10;
    unsigned int flip_passes = 10;
    unsigned int smooth_sub_iterations = 5;

    // CUDA tuning
    int cuda_block_size = 256;
    int gpu_capacity_multiplier = 4;
    int bvh_max_leaf_size = 8;

    // Checkpoint control
    bool checkpoint_enabled = false;
    std::string checkpoint_dir;
    bool checkpoint_save_all = false;
    std::string checkpoint_save_at;   // specific stage name, or empty
    std::string checkpoint_run_from;  // resume from this stage
    std::string checkpoint_run_to;    // stop after this stage
};

// Load config from plain-text key-value file (QuadWild pattern)
RemeshingCudaParams load_remeshing_config(const std::string& path);

// Save config (for debugging / reproducibility)
void save_remeshing_config(const std::string& path,
                           const RemeshingCudaParams& params);

// CUDA-accelerated uniform remeshing with config-driven strategy dispatch
void uniform_remeshing_cuda(SurfaceMesh& mesh, Scalar edge_length,
                            const RemeshingCudaParams& params,
                            bool use_projection = true);

// CUDA-accelerated adaptive remeshing with config-driven strategy dispatch
void adaptive_remeshing_cuda(SurfaceMesh& mesh, Scalar min_edge_length,
                             Scalar max_edge_length, Scalar approx_error,
                             const RemeshingCudaParams& params,
                             bool use_projection = true);

} // namespace pmp
