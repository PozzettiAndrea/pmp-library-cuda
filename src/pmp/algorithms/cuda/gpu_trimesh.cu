// SPDX-License-Identifier: MIT

#include "gpu_trimesh.cuh"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[GPU] CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

namespace pmp {

void gpu_trimesh_alloc(GpuTriMesh& gm, int capV, int capF)
{
    gpu_trimesh_free(gm);

    gm.capV = capV;
    gm.capF = capF;

    // Geometry
    CUDA_CHECK(cudaMalloc(&gm.d_V, 3 * capV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm.d_vnormal, 3 * capV * sizeof(float)));

    // Connectivity
    CUDA_CHECK(cudaMalloc(&gm.d_F, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gm.d_E2E, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gm.d_vhalfedge, capV * sizeof(int)));

    // Per-vertex properties
    CUDA_CHECK(cudaMalloc(&gm.d_vsizing, capV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gm.d_vlocked, capV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gm.d_vfeature, capV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gm.d_vboundary, capV * sizeof(int)));

    // Per-halfedge properties
    CUDA_CHECK(cudaMalloc(&gm.d_efeature, 3 * capF * sizeof(int)));

    // Temporary buffers
    CUDA_CHECK(cudaMalloc(&gm.d_update, 3 * capV * sizeof(float)));

    // Zero everything
    CUDA_CHECK(cudaMemset(gm.d_V, 0, 3 * capV * sizeof(float)));
    CUDA_CHECK(cudaMemset(gm.d_vnormal, 0, 3 * capV * sizeof(float)));
    CUDA_CHECK(cudaMemset(gm.d_F, 0, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_E2E, 0xFF, 3 * capF * sizeof(int))); // -1
    CUDA_CHECK(cudaMemset(gm.d_vhalfedge, 0xFF, capV * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_vsizing, 0, capV * sizeof(float)));
    CUDA_CHECK(cudaMemset(gm.d_vlocked, 0, capV * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_vfeature, 0, capV * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_vboundary, 0, capV * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_efeature, 0, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMemset(gm.d_update, 0, 3 * capV * sizeof(float)));

    printf("[GPU] Allocated GpuTriMesh: capV=%d capF=%d (%.1f MB)\n", capV,
           capF,
           (3 * capV * sizeof(float) * 3 + 3 * capF * sizeof(int) * 2 +
            capV * sizeof(int) * 4 + capV * sizeof(float) +
            3 * capF * sizeof(int)) /
               (1024.0 * 1024.0));
}

void gpu_trimesh_free(GpuTriMesh& gm)
{
    if (gm.d_V) cudaFree(gm.d_V);
    if (gm.d_vnormal) cudaFree(gm.d_vnormal);
    if (gm.d_F) cudaFree(gm.d_F);
    if (gm.d_E2E) cudaFree(gm.d_E2E);
    if (gm.d_vhalfedge) cudaFree(gm.d_vhalfedge);
    if (gm.d_vsizing) cudaFree(gm.d_vsizing);
    if (gm.d_vlocked) cudaFree(gm.d_vlocked);
    if (gm.d_vfeature) cudaFree(gm.d_vfeature);
    if (gm.d_vboundary) cudaFree(gm.d_vboundary);
    if (gm.d_efeature) cudaFree(gm.d_efeature);
    if (gm.d_update) cudaFree(gm.d_update);

    gm = GpuTriMesh{};
}

void gpu_trimesh_resize(GpuTriMesh& gm, int nV, int nF, int capacity_mult)
{
    if (nV <= gm.capV && nF <= gm.capF)
    {
        gm.nV = nV;
        gm.nF = nF;
        return;
    }

    int newCapV = nV * capacity_mult;
    int newCapF = nF * capacity_mult;
    if (newCapV < gm.capV) newCapV = gm.capV;
    if (newCapF < gm.capF) newCapF = gm.capF;

    printf("[GPU] Resizing GpuTriMesh: %d/%d -> %d/%d (cap %d/%d)\n", gm.nV,
           gm.nF, nV, nF, newCapV, newCapF);

    gpu_trimesh_alloc(gm, newCapV, newCapF);
    gm.nV = nV;
    gm.nF = nF;
}

} // namespace pmp
