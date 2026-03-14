// SPDX-License-Identifier: MIT

// Host-side orchestration for CUDA tangential smoothing.
// Takes flat arrays, uploads to GPU, runs kernels, downloads results.

#include "gpu_trimesh.cuh"
#include "gpu_bvh.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[SMOOTH-HOST] CUDA error at %s:%d: %s\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err));                          \
        }                                                                      \
    } while (0)

// Check if any CUDA device is available
extern "C" bool pmp_cuda_device_available()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return false;
    // Try to set device 0
    err = cudaSetDevice(0);
    return err == cudaSuccess;
}

namespace pmp {

// Forward declarations from remesh_smooth.cu
extern void cuda_compute_vertex_normals(GpuTriMesh& gm);
extern void cuda_tangential_smooth_iteration(GpuTriMesh& gm);
extern void cuda_project_to_reference(GpuTriMesh& gm, const GpuBVH& bvh,
                                      const float* d_ref_vnormal,
                                      const float* d_ref_vsizing,
                                      const int* d_ref_F, int ref_nV);

// ============================================================
// cuda_smooth_pipeline: upload, smooth N iterations, download
// ============================================================

extern "C" void cuda_smooth_pipeline(
    // Working mesh (flat arrays, modified in place)
    float* positions, float* normals, const int* faces, const int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature, int nV,
    int nF,
    // Reference mesh for projection (can be null)
    const float* ref_positions, const float* ref_normals,
    const float* ref_sizing, const int* ref_faces, int ref_nV, int ref_nF,
    // Parameters
    int smooth_iterations, int use_projection, int block_size,
    int capacity_mult)
{
    (void)block_size; // used implicitly in kernels (fixed at 256)

    // Allocate and upload working mesh
    GpuTriMesh gm;
    gpu_trimesh_alloc(gm, nV * capacity_mult, nF * capacity_mult);
    gm.nV = nV;
    gm.nF = nF;

    CUDA_CHECK(cudaMemcpy(gm.d_V, positions, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vnormal, normals, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_F, faces, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_E2E, e2e, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vhalfedge, vhalfedge, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vsizing, vsizing, nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vlocked, vlocked, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vfeature, vfeature, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vboundary, vboundary, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_efeature, efeature, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));

    // Build BVH from reference mesh if projection is enabled
    GpuBVH bvh;
    float* d_ref_vnormal = nullptr;
    float* d_ref_vsizing = nullptr;
    int* d_ref_F = nullptr;

    if (use_projection && ref_positions && ref_nF > 0)
    {
        // Upload reference mesh vertex positions and faces to build BVH
        float* d_ref_V;
        CUDA_CHECK(cudaMalloc(&d_ref_V, 3 * ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ref_V, ref_positions, 3 * ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_ref_F, 3 * ref_nF * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ref_F, ref_faces, 3 * ref_nF * sizeof(int), cudaMemcpyHostToDevice));

        gpu_bvh_build(bvh, d_ref_V, d_ref_F, ref_nF, ref_nV);

        // Upload reference normals and sizing
        CUDA_CHECK(cudaMalloc(&d_ref_vnormal, 3 * ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ref_vnormal, ref_normals, 3 * ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_ref_vsizing, ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ref_vsizing, ref_sizing, ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        cudaFree(d_ref_V);

        // Project at the beginning
        cuda_project_to_reference(gm, bvh, d_ref_vnormal, d_ref_vsizing,
                                  d_ref_F, ref_nV);
    }

    // Run smoothing iterations
    for (int iter = 0; iter < smooth_iterations; ++iter)
    {
        cuda_tangential_smooth_iteration(gm);
    }

    // Project at the end
    if (use_projection && bvh.nTriangles > 0)
    {
        cuda_project_to_reference(gm, bvh, d_ref_vnormal, d_ref_vsizing,
                                  d_ref_F, ref_nV);
    }

    // Download results back to host
    CUDA_CHECK(cudaMemcpy(positions, gm.d_V, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(normals, gm.d_vnormal, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vsizing, gm.d_vsizing, nV * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    if (d_ref_vnormal) cudaFree(d_ref_vnormal);
    if (d_ref_vsizing) cudaFree(d_ref_vsizing);
    if (d_ref_F) cudaFree(d_ref_F);
    gpu_bvh_free(bvh);
    gpu_trimesh_free(gm);
}

// ============================================================
// Persistent smooth API: build BVH once, reuse across iterations
// ============================================================

struct SmoothPersistentState
{
    GpuTriMesh gm;
    GpuBVH bvh;
    float* d_ref_vnormal = nullptr;
    float* d_ref_vsizing = nullptr;
    int* d_ref_F = nullptr;
    int ref_nV = 0;
    int capacity_mult = 4;
    bool has_projection = false;
};

extern "C" void cuda_smooth_init(
    const float* ref_positions, const float* ref_normals,
    const float* ref_sizing, const int* ref_faces,
    int ref_nV, int ref_nF, int capacity_mult, void** handle_out)
{
    auto* state = new SmoothPersistentState();
    state->capacity_mult = capacity_mult;
    state->ref_nV = ref_nV;

    if (ref_positions && ref_nF > 0)
    {
        state->has_projection = true;

        // Upload reference mesh and build BVH (once)
        float* d_ref_V;
        CUDA_CHECK(cudaMalloc(&d_ref_V, 3 * ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ref_V, ref_positions, 3 * ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&state->d_ref_F, 3 * ref_nF * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(state->d_ref_F, ref_faces, 3 * ref_nF * sizeof(int), cudaMemcpyHostToDevice));

        gpu_bvh_build(state->bvh, d_ref_V, state->d_ref_F, ref_nF, ref_nV);
        cudaFree(d_ref_V);

        CUDA_CHECK(cudaMalloc(&state->d_ref_vnormal, 3 * ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(state->d_ref_vnormal, ref_normals, 3 * ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&state->d_ref_vsizing, ref_nV * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(state->d_ref_vsizing, ref_sizing, ref_nV * sizeof(float), cudaMemcpyHostToDevice));

        printf("[SMOOTH-PERSIST] BVH built: %d tris, ref uploaded: %d verts\n",
               ref_nF, ref_nV);
    }

    *handle_out = state;
}

extern "C" void cuda_smooth_iteration(
    void* handle,
    float* positions, float* normals, const int* faces, const int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature,
    int nV, int nF, int smooth_iterations, int use_projection)
{
    auto* state = (SmoothPersistentState*)handle;

    // Resize GpuTriMesh if needed (reallocs only when capacity exceeded)
    gpu_trimesh_resize(state->gm, nV, nF, state->capacity_mult);

    // Upload working mesh arrays
    CUDA_CHECK(cudaMemcpy(state->gm.d_V, positions, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vnormal, normals, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_F, faces, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_E2E, e2e, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vhalfedge, vhalfedge, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vsizing, vsizing, nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vlocked, vlocked, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vfeature, vfeature, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vboundary, vboundary, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_efeature, efeature, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));

    // Project at beginning
    if (use_projection && state->has_projection)
    {
        cuda_project_to_reference(state->gm, state->bvh, state->d_ref_vnormal,
                                  state->d_ref_vsizing, state->d_ref_F,
                                  state->ref_nV);
    }

    // Run smooth sub-iterations
    for (int i = 0; i < smooth_iterations; ++i)
        cuda_tangential_smooth_iteration(state->gm);

    // Project at end
    if (use_projection && state->has_projection)
    {
        cuda_project_to_reference(state->gm, state->bvh, state->d_ref_vnormal,
                                  state->d_ref_vsizing, state->d_ref_F,
                                  state->ref_nV);
    }

    // Download only positions, normals, sizing
    CUDA_CHECK(cudaMemcpy(positions, state->gm.d_V, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(normals, state->gm.d_vnormal, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vsizing, state->gm.d_vsizing, nV * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================
// Fused GPU phase: normals + flip + smooth in one shot
// Downloads V + F + E2E for full mesh rebuild
// ============================================================

// Forward declaration from remesh_flip.cu
extern int cuda_flip_edges(GpuTriMesh& gm, int max_passes);

extern "C" void cuda_fused_flip_smooth(
    void* handle,
    // Working mesh (flat arrays — positions/vsizing modified in place,
    // faces/e2e modified in place if flips occur)
    float* positions, float* normals, int* faces, int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature,
    int nV, int nF,
    int flip_passes, int smooth_iterations, int use_projection,
    int* out_flips)
{
    auto* state = (SmoothPersistentState*)handle;

    // Resize GpuTriMesh if needed
    gpu_trimesh_resize(state->gm, nV, nF, state->capacity_mult);

    // Upload all working mesh arrays
    CUDA_CHECK(cudaMemcpy(state->gm.d_V, positions, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vnormal, normals, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_F, faces, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_E2E, e2e, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vhalfedge, vhalfedge, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vsizing, vsizing, nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vlocked, vlocked, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vfeature, vfeature, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_vboundary, vboundary, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->gm.d_efeature, efeature, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));

    // 1. GPU normals (replaces CPU vertex_normals call)
    cuda_compute_vertex_normals(state->gm);

    // 2. GPU flip (modifies d_F, d_E2E, d_vhalfedge on device)
    int n_flips = 0;
    if (flip_passes > 0)
        n_flips = cuda_flip_edges(state->gm, flip_passes);
    if (out_flips) *out_flips = n_flips;

    // 3. GPU smooth (modifies d_V, d_vnormal, d_vsizing on device)
    // Project at beginning
    if (use_projection && state->has_projection)
    {
        cuda_project_to_reference(state->gm, state->bvh, state->d_ref_vnormal,
                                  state->d_ref_vsizing, state->d_ref_F,
                                  state->ref_nV);
    }

    for (int i = 0; i < smooth_iterations; ++i)
        cuda_tangential_smooth_iteration(state->gm);

    // Project at end
    if (use_projection && state->has_projection)
    {
        cuda_project_to_reference(state->gm, state->bvh, state->d_ref_vnormal,
                                  state->d_ref_vsizing, state->d_ref_F,
                                  state->ref_nV);
    }

    // Download everything needed for mesh rebuild
    CUDA_CHECK(cudaMemcpy(positions, state->gm.d_V, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(normals, state->gm.d_vnormal, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vsizing, state->gm.d_vsizing, nV * sizeof(float), cudaMemcpyDeviceToHost));

    // Download F and E2E only if flips occurred (topology changed)
    if (n_flips > 0)
    {
        CUDA_CHECK(cudaMemcpy(faces, state->gm.d_F, 3 * nF * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(e2e, state->gm.d_E2E, 3 * nF * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

extern "C" void cuda_smooth_cleanup(void* handle)
{
    auto* state = (SmoothPersistentState*)handle;
    if (!state) return;

    gpu_trimesh_free(state->gm);
    gpu_bvh_free(state->bvh);
    if (state->d_ref_vnormal) cudaFree(state->d_ref_vnormal);
    if (state->d_ref_vsizing) cudaFree(state->d_ref_vsizing);
    if (state->d_ref_F) cudaFree(state->d_ref_F);

    delete state;
    printf("[SMOOTH-PERSIST] Cleaned up persistent state\n");
}

} // namespace pmp
