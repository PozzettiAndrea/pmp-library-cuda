// SPDX-License-Identifier: MIT
#pragma once

// Host-side bridge between SurfaceMesh and CUDA flat arrays.
// This file can include PMP headers (compiled as C++20).
// CUDA kernels are called via extern C-style functions declared here.

#include "pmp/surface_mesh.h"

namespace pmp {

struct GpuTriMesh;
struct GpuBVH;

// ============================================================
// Upload SurfaceMesh to flat arrays (host-side extraction)
// Returns flat arrays that can be cudaMemcpy'd to device
// ============================================================

struct FlatTriMesh
{
    std::vector<float> positions;    // [3*nV]
    std::vector<float> normals;      // [3*nV]
    std::vector<int> faces;          // [3*nF]
    std::vector<int> e2e;            // [3*nF] half-edge twins
    std::vector<int> vhalfedge;      // [nV] outgoing half-edge per vertex
    std::vector<float> vsizing;      // [nV]
    std::vector<int> vlocked;        // [nV]
    std::vector<int> vfeature;       // [nV]
    std::vector<int> vboundary;      // [nV]
    std::vector<int> efeature;       // [3*nF] per-halfedge
    int nV = 0, nF = 0;
};

// Extract flat arrays from SurfaceMesh
FlatTriMesh extract_flat_mesh(const SurfaceMesh& mesh);

// Restore SurfaceMesh positions from flat arrays (positions only, no topology change)
void restore_positions(SurfaceMesh& mesh, const std::vector<float>& positions);

// Restore full mesh from flat arrays
void restore_full_mesh(SurfaceMesh& mesh, const FlatTriMesh& flat);

// ============================================================
// CUDA smooth entry point (C++ wrapper calling CUDA kernels)
// ============================================================

#ifdef PMP_HAS_CUDA
// Upload flat mesh to GPU, run smoothing iterations, download positions back
void cuda_smooth_on_gpu(SurfaceMesh& mesh, unsigned int smooth_iterations,
                        bool use_projection, int block_size,
                        int capacity_mult);
#endif

} // namespace pmp
