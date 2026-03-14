// SPDX-License-Identifier: MIT
#pragma once

// ============================================================
// Flat GPU triangle mesh representation
//
// Derived from SurfaceMesh's half-edge structure for GPU use.
// Uses F[3*nF] + E2E[3*nF] layout (same as QuadriFlow-cuda's
// subdivide_gpu.cu pattern).
//
// One-ring traversal on GPU:
//   next(he) = 3*(he/3) + (he+1)%3
//   prev(he) = 3*(he/3) + (he+2)%3
//   twin(he) = E2E[he]  (-1 = boundary)
// ============================================================

#include <cstdint>

namespace pmp {

struct GpuTriMesh
{
    // Geometry
    float* d_V = nullptr;          // [3*nV] vertex positions (x,y,z interleaved)
    float* d_vnormal = nullptr;    // [3*nV] vertex normals

    // Connectivity (triangle-based)
    int* d_F = nullptr;            // [3*nF] face-vertex indices
    int* d_E2E = nullptr;          // [3*nF] half-edge twin indices (-1 = boundary)
    int* d_vhalfedge = nullptr;    // [nV] one outgoing half-edge per vertex

    // Per-vertex properties
    float* d_vsizing = nullptr;    // [nV] target edge length
    int* d_vlocked = nullptr;      // [nV] 0/1
    int* d_vfeature = nullptr;     // [nV] 0/1
    int* d_vboundary = nullptr;    // [nV] 0/1

    // Per-halfedge properties
    int* d_efeature = nullptr;     // [3*nF] feature flag per half-edge

    // Temporary buffers for kernels
    float* d_update = nullptr;     // [3*nV] smooth update vectors

    int nV = 0, nF = 0;           // current counts
    int capV = 0, capF = 0;       // allocated capacity
};

// Allocate GPU mesh with given capacity
void gpu_trimesh_alloc(GpuTriMesh& gm, int capV, int capF);

// Free GPU mesh memory
void gpu_trimesh_free(GpuTriMesh& gm);

// Resize (reallocate if needed, preserving capacity multiplier)
void gpu_trimesh_resize(GpuTriMesh& gm, int nV, int nF, int capacity_mult);

} // namespace pmp
