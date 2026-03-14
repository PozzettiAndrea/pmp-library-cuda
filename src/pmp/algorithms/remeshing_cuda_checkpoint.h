// SPDX-License-Identifier: MIT
#pragma once

// ============================================================
// Pipeline checkpoint system for PMP CUDA remeshing
//
// Modeled after QuadriFlow-cuda and QuadWild-BiMDF checkpoint
// systems. Allows saving/loading pipeline state at any stage
// boundary for fast benchmarking and resumption.
//
// Usage:
//   Full run with saves:
//     uniform_remeshing_cuda(mesh, 0.1, params)
//     // with params.checkpoint_save_all = true,
//     //      params.checkpoint_dir = "/tmp/ckpt"
//
//   Resume from stage:
//     params.checkpoint_run_from = "post-split"
//     uniform_remeshing_cuda(mesh, 0.1, params)
//
// Stage names (in pipeline order, repeated per iteration):
//   preprocessed    After sizing field + reference mesh/BVH built
//   post-split      After split_long_edges
//   post-collapse   After collapse_short_edges + compaction
//   post-flip       After flip_edges
//   post-smooth     After tangential_smoothing
//   complete        After remove_caps + postprocessing
// ============================================================

#include <cstdint>
#include <string>
#include <vector>

namespace pmp {

enum RemeshStage
{
    REMESH_STAGE_NONE = -1,
    REMESH_STAGE_PREPROCESSED = 0,
    REMESH_STAGE_POST_SPLIT,
    REMESH_STAGE_POST_COLLAPSE,
    REMESH_STAGE_POST_FLIP,
    REMESH_STAGE_POST_SMOOTH,
    REMESH_STAGE_COMPLETE,
    REMESH_STAGE_COUNT
};

RemeshStage remesh_stage_from_name(const char* name);
const char* remesh_stage_name(RemeshStage s);
void remesh_list_stages();

struct RemeshCheckpointHeader
{
    char magic[4];            // "RMC\0"
    int32_t version;          // format version (1)
    char stage[64];           // stage name string
    int32_t iteration;        // current outer iteration (-1 for pre/post)
    int32_t num_vertices;
    int32_t num_faces;
    float target_edge_length; // uniform target (0 for adaptive)
    float min_edge_length;    // adaptive min
    float max_edge_length;    // adaptive max
    float approx_error;       // adaptive approx error
    int32_t is_uniform;       // 1 = uniform, 0 = adaptive
    int32_t use_projection;
    // Strategy flags for reproducibility
    int32_t split_strategy;
    int32_t collapse_strategy;
    int32_t flip_strategy;
    int32_t smooth_strategy;
    int64_t timestamp;
    char reserved[128];
};

// Mesh data for checkpoint serialization (flat arrays extracted from SurfaceMesh)
struct RemeshCheckpointData
{
    // Vertex positions (x,y,z interleaved)
    std::vector<float> positions;
    // Face vertex indices (v0,v1,v2 interleaved for triangles)
    std::vector<int32_t> faces;

    // Per-vertex properties
    std::vector<float> vsizing;     // target edge length per vertex
    std::vector<int32_t> vlocked;   // 0/1
    std::vector<int32_t> vfeature;  // 0/1

    // Per-edge properties (indexed by edge index)
    std::vector<int32_t> efeature;  // 0/1
    std::vector<int32_t> elocked;   // 0/1

    // Reference mesh for projection (saved at preprocessed stage)
    bool has_reference = false;
    std::vector<float> ref_positions;
    std::vector<int32_t> ref_faces;
    std::vector<float> ref_normals;   // per-vertex normals
    std::vector<float> ref_sizing;    // per-vertex sizing
};

// Save checkpoint data to directory
void save_remesh_checkpoint(const RemeshCheckpointData& data, RemeshStage stage,
                            const char* dir, const RemeshCheckpointHeader& hdr);

// Load checkpoint, returns stage that was saved
RemeshStage load_remesh_checkpoint(RemeshCheckpointData& data, const char* dir,
                                   RemeshStage stage);

// Check if a checkpoint file exists for a given stage
bool remesh_checkpoint_exists(const char* dir, RemeshStage stage);

} // namespace pmp
