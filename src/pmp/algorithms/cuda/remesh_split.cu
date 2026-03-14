// SPDX-License-Identifier: MIT

// ============================================================
// CUDA kernels for edge splitting
//
// Port from QuadriFlow-cuda's subdivide_gpu.cu with PMP modifications:
//   - Per-vertex sizing threshold: 4/3 * min(vsizing[v0], vsizing[v1])
//   - Feature edge/vertex propagation
//   - Sizing interpolation: vsizing[vnew] = 0.5*(vsizing[v0]+vsizing[v1])
//   - float instead of double
//
// Pattern: mark → resolve within-face → resolve cross-face →
//          resolve neighbor → scan → apply splits → rebuild E2E
// ============================================================

#include "gpu_trimesh.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[SPLIT] CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

namespace pmp {

// ============================================================
// Kernel: mark long edges
// ============================================================

__global__ void k_mark_long_edges_pmp(const float* __restrict__ V,
                                      const int* __restrict__ F,
                                      const int* __restrict__ E2E,
                                      const float* __restrict__ vsizing,
                                      const int* __restrict__ efeature,
                                      int nHE,
                                      int* __restrict__ edge_marks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHE) return;

    edge_marks[idx] = 0;

    int twin = E2E[idx];
    // Canonical: only mark if boundary (twin==-1) or idx < twin
    if (twin != -1 && idx > twin) return;

    // Don't split locked edges (we use efeature for locked too — feature edges
    // CAN be split, but their sub-edges inherit the feature flag)
    // Actually PMP splits feature edges too, so no skip here.

    int f = idx / 3, j = idx % 3;
    int v0 = F[3 * f + j];
    int v1 = F[3 * f + (j + 1) % 3];

    float s0 = vsizing[v0], s1 = vsizing[v1];
    float threshold = (4.0f / 3.0f) * fminf(s0, s1);

    float dx = V[3 * v0] - V[3 * v1];
    float dy = V[3 * v0 + 1] - V[3 * v1 + 1];
    float dz = V[3 * v0 + 2] - V[3 * v1 + 2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if (dist > threshold)
    {
        edge_marks[idx] = 1;
    }
}

// ============================================================
// Kernel: resolve within-face conflicts (keep longest)
// ============================================================

__global__ void k_resolve_within_face_pmp(const float* __restrict__ V,
                                          const int* __restrict__ F, int nF,
                                          int* __restrict__ edge_marks)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;

    int count = edge_marks[3 * f] + edge_marks[3 * f + 1] + edge_marks[3 * f + 2];
    if (count <= 1) return;

    float best_len = -1.0f;
    int best_j = -1;
    for (int j = 0; j < 3; ++j)
    {
        if (!edge_marks[3 * f + j]) continue;
        int v0 = F[3 * f + j], v1 = F[3 * f + (j + 1) % 3];
        float dx = V[3 * v0] - V[3 * v1];
        float dy = V[3 * v0 + 1] - V[3 * v1 + 1];
        float dz = V[3 * v0 + 2] - V[3 * v1 + 2];
        float len = dx * dx + dy * dy + dz * dz;
        if (len > best_len) { best_len = len; best_j = j; }
    }
    for (int j = 0; j < 3; ++j)
        if (j != best_j) edge_marks[3 * f + j] = 0;
}

// ============================================================
// Kernel: resolve cross-face conflicts (lower index wins)
// ============================================================

__global__ void k_resolve_cross_face_pmp(const int* __restrict__ E2E, int nHE,
                                         int* __restrict__ edge_marks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHE) return;
    if (!edge_marks[idx]) return;

    int other = E2E[idx];
    if (other != -1 && edge_marks[other] && idx > other)
        edge_marks[idx] = 0;
}

// ============================================================
// Kernel: resolve neighbor conflicts (Luby-like)
// ============================================================

__global__ void k_resolve_neighbor_pmp(const float* __restrict__ V,
                                       const int* __restrict__ F,
                                       const int* __restrict__ E2E, int nHE,
                                       int* __restrict__ edge_marks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHE) return;
    if (!edge_marks[idx]) return;

    int f0 = idx / 3, j0 = idx % 3;
    int va = F[3 * f0 + j0], vb = F[3 * f0 + (j0 + 1) % 3];
    float dx = V[3 * va] - V[3 * vb], dy = V[3 * va + 1] - V[3 * vb + 1],
          dz = V[3 * va + 2] - V[3 * vb + 2];
    float my_len = dx * dx + dy * dy + dz * dz;

    auto dominated = [&](int oi) -> bool {
        if (oi < 0 || !edge_marks[oi]) return false;
        int fo = oi / 3, jo = oi % 3;
        int ua = F[3 * fo + jo], ub = F[3 * fo + (jo + 1) % 3];
        float ex = V[3 * ua] - V[3 * ub], ey = V[3 * ua + 1] - V[3 * ub + 1],
              ez = V[3 * ua + 2] - V[3 * ub + 2];
        float ol = ex * ex + ey * ey + ez * ez;
        return (ol > my_len) || (ol == my_len && oi > idx);
    };

    // Check face 0
    for (int j = 0; j < 3; ++j)
    {
        int he = 3 * f0 + j;
        if (he == idx) continue;
        if (dominated(E2E[he])) { edge_marks[idx] = 0; return; }
    }

    // Check face 1 (twin's face)
    int twin = E2E[idx];
    if (twin == -1) return;
    int f1 = twin / 3;
    for (int j = 0; j < 3; ++j)
    {
        int he = 3 * f1 + j;
        if (he == twin) continue;
        if (dominated(he)) { edge_marks[idx] = 0; return; }
        if (dominated(E2E[he])) { edge_marks[idx] = 0; return; }
    }
}

// ============================================================
// Kernel: compute face counts per marked edge
// ============================================================

__global__ void k_face_counts_pmp(const int* __restrict__ edge_marks,
                                  const int* __restrict__ E2E, int nHE,
                                  int* __restrict__ face_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHE) return;
    if (!edge_marks[idx]) { face_counts[idx] = 0; return; }
    face_counts[idx] = (E2E[idx] == -1) ? 1 : 2;
}

// ============================================================
// Kernel: apply splits
// ============================================================

__global__ void k_apply_splits_pmp(
    const int* __restrict__ F_old, int* __restrict__ F, float* __restrict__ V,
    float* __restrict__ vsizing, int* __restrict__ vfeature,
    int* __restrict__ vboundary, const int* __restrict__ efeature,
    const int* __restrict__ E2E, const int* __restrict__ edge_marks,
    const int* __restrict__ vtx_scan, const int* __restrict__ face_scan,
    int nV_old, int nF_old, int nHE_old)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHE_old) return;
    if (!edge_marks[idx]) return;

    int f0 = idx / 3, j0 = idx % 3;
    int e1 = E2E[idx];
    bool is_bnd = (e1 == -1);

    // Read from snapshot
    int v0 = F_old[3 * f0 + j0];
    int v0p = F_old[3 * f0 + (j0 + 2) % 3];
    int v1 = F_old[3 * f0 + (j0 + 1) % 3];

    int vn = nV_old + vtx_scan[idx];

    // New vertex = midpoint
    V[3 * vn] = 0.5f * (V[3 * v0] + V[3 * v1]);
    V[3 * vn + 1] = 0.5f * (V[3 * v0 + 1] + V[3 * v1 + 1]);
    V[3 * vn + 2] = 0.5f * (V[3 * v0 + 2] + V[3 * v1 + 2]);

    // Interpolate sizing
    vsizing[vn] = 0.5f * (vsizing[v0] + vsizing[v1]);

    // Feature propagation
    vfeature[vn] = efeature[idx] ? 1 : 0;
    vboundary[vn] = is_bnd ? 1 : 0;

    // Rewrite f0: (vn, v0p, v0)
    F[3 * f0 + 0] = vn;
    F[3 * f0 + 1] = v0p;
    F[3 * f0 + 2] = v0;

    // New face f3: (vn, v1, v0p)
    int f3 = nF_old + face_scan[idx];
    F[3 * f3 + 0] = vn;
    F[3 * f3 + 1] = v1;
    F[3 * f3 + 2] = v0p;

    if (!is_bnd)
    {
        int f1 = e1 / 3, j1 = e1 % 3;
        int v1p = F_old[3 * f1 + (j1 + 2) % 3];

        // Rewrite f1: (vn, v0, v1p)
        F[3 * f1 + 0] = vn;
        F[3 * f1 + 1] = v0;
        F[3 * f1 + 2] = v1p;

        // New face f2: (vn, v1p, v1)
        int f2 = nF_old + face_scan[idx] + 1;
        F[3 * f2 + 0] = vn;
        F[3 * f2 + 1] = v1p;
        F[3 * f2 + 2] = v1;
    }
}

// ============================================================
// Kernel: build E2E sort keys
// ============================================================

__global__ void k_build_e2e_keys_pmp(const int* __restrict__ F, int nF,
                                     int maxV, long long* __restrict__ keys,
                                     int* __restrict__ indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;

    int f = idx / 3, j = idx % 3;
    int va = F[3 * f + j], vb = F[3 * f + (j + 1) % 3];
    long long lo = (va < vb) ? va : vb;
    long long hi = (va < vb) ? vb : va;
    keys[idx] = lo * (long long)maxV + hi;
    indices[idx] = idx;
}

// ============================================================
// Kernel: pair sorted half-edges for E2E
// ============================================================

__global__ void k_pair_e2e_pmp(const long long* __restrict__ sorted_keys,
                               const int* __restrict__ sorted_indices, int nHE,
                               int* __restrict__ E2E)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nHE) return;

    long long my_key = sorted_keys[i];
    bool is_first = (i == 0 || sorted_keys[i - 1] != my_key);
    bool has_next = (i + 1 < nHE && sorted_keys[i + 1] == my_key);

    if (is_first && has_next)
    {
        int h0 = sorted_indices[i];
        int h1 = sorted_indices[i + 1];
        E2E[h0] = h1;
        E2E[h1] = h0;
    }
}

// ============================================================
// Host: GPU-resident split pipeline
// ============================================================

extern "C" void cuda_split_pipeline(
    float* h_V, int nV_in, int* h_F, int nF_in, float* h_vsizing,
    int* h_vfeature, int* h_vboundary, int* h_efeature, int max_passes,
    int capacity_mult,
    // Outputs (caller must free with free())
    float** V_out, int* nV_out, int** F_out, int* nF_out,
    float** vsizing_out, int** vfeature_out, int** vboundary_out)
{
    int nV = nV_in, nF = nF_in;
    int capV = nV * capacity_mult;
    int capF = nF * capacity_mult;
    int capHE = capF * 3;

    // Allocate GPU arrays
    float *d_V, *d_vsizing;
    int *d_F, *d_E2E, *d_vfeature, *d_vboundary, *d_efeature;
    CUDA_CHECK(cudaMalloc(&d_V, 3 * capV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_F, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_E2E, capHE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vsizing, capV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vfeature, capV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vboundary, capV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_efeature, capHE * sizeof(int)));

    // Work arrays
    int *d_marks, *d_face_counts, *d_vtx_scan, *d_face_scan, *d_F_old;
    long long *d_sort_keys;
    int *d_sort_indices;
    CUDA_CHECK(cudaMalloc(&d_marks, capHE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_face_counts, capHE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vtx_scan, capHE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_face_scan, capHE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_F_old, 3 * capF * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sort_keys, capHE * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sort_indices, capHE * sizeof(int)));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_V, h_V, 3 * nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vsizing, h_vsizing, nV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vfeature, h_vfeature, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vboundary, h_vboundary, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_efeature, h_efeature, 3 * nF * sizeof(int), cudaMemcpyHostToDevice));

    // Build initial E2E via sort
    {
        int nHE = 3 * nF;
        int maxV = nV + 1;
        int grid = (nHE + 255) / 256;
        CUDA_CHECK(cudaMemset(d_E2E, 0xFF, nHE * sizeof(int)));
        k_build_e2e_keys_pmp<<<grid, 256>>>(d_F, nF, maxV, d_sort_keys, d_sort_indices);
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<long long> dp_keys(d_sort_keys);
        thrust::device_ptr<int> dp_idx(d_sort_indices);
        thrust::sort_by_key(thrust::device, dp_keys, dp_keys + nHE, dp_idx);
        k_pair_e2e_pmp<<<grid, 256>>>(d_sort_keys, d_sort_indices, nHE, d_E2E);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const int BS = 256;
    int total_splits = 0;

    for (int pass = 0; pass < max_passes; ++pass)
    {
        int nHE = 3 * nF;
        int gridHE = (nHE + BS - 1) / BS;
        int gridF = (nF + BS - 1) / BS;

        // Mark long edges
        CUDA_CHECK(cudaMemset(d_marks, 0, nHE * sizeof(int)));
        k_mark_long_edges_pmp<<<gridHE, BS>>>(d_V, d_F, d_E2E, d_vsizing,
                                              d_efeature, nHE, d_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Resolve within-face
        k_resolve_within_face_pmp<<<gridF, BS>>>(d_V, d_F, nF, d_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Resolve cross-face
        k_resolve_cross_face_pmp<<<gridHE, BS>>>(d_E2E, nHE, d_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Resolve neighbor conflicts
        k_resolve_neighbor_pmp<<<gridHE, BS>>>(d_V, d_F, d_E2E, nHE, d_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Count splits
        thrust::device_ptr<int> dp_marks(d_marks);
        int n_splits = thrust::reduce(thrust::device, dp_marks, dp_marks + nHE);
        if (n_splits == 0) break;

        // Compute face counts
        k_face_counts_pmp<<<gridHE, BS>>>(d_marks, d_E2E, nHE, d_face_counts);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exclusive scans
        thrust::device_ptr<int> dp_vtx(d_vtx_scan);
        thrust::device_ptr<int> dp_fc(d_face_counts);
        thrust::device_ptr<int> dp_fs(d_face_scan);

        // vtx_scan = exclusive_scan(marks)
        CUDA_CHECK(cudaMemcpy(d_vtx_scan, d_marks, nHE * sizeof(int), cudaMemcpyDeviceToDevice));
        thrust::exclusive_scan(thrust::device, dp_marks, dp_marks + nHE, dp_vtx);

        // face_scan = exclusive_scan(face_counts)
        thrust::exclusive_scan(thrust::device, dp_fc, dp_fc + nHE, dp_fs);

        // Get total new faces
        int last_fc, last_fs;
        CUDA_CHECK(cudaMemcpy(&last_fc, d_face_counts + nHE - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_fs, d_face_scan + nHE - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int new_faces = last_fs + last_fc;

        int new_nV = nV + n_splits;
        int new_nF = nF + new_faces;

        // Check capacity
        if (new_nV > capV || new_nF > capF)
        {
            printf("[SPLIT] Capacity exceeded: nV=%d/%d nF=%d/%d, stopping\n",
                   new_nV, capV, new_nF, capF);
            break;
        }

        // Snapshot F
        CUDA_CHECK(cudaMemcpy(d_F_old, d_F, 3 * nF * sizeof(int), cudaMemcpyDeviceToDevice));

        // Apply splits
        k_apply_splits_pmp<<<gridHE, BS>>>(
            d_F_old, d_F, d_V, d_vsizing, d_vfeature, d_vboundary,
            d_efeature, d_E2E, d_marks, d_vtx_scan, d_face_scan, nV, nF, nHE);
        CUDA_CHECK(cudaDeviceSynchronize());

        nV = new_nV;
        nF = new_nF;
        total_splits += n_splits;

        // Rebuild E2E for new mesh
        nHE = 3 * nF;
        int maxV = nV + 1;
        gridHE = (nHE + BS - 1) / BS;
        CUDA_CHECK(cudaMemset(d_E2E, 0xFF, nHE * sizeof(int)));
        k_build_e2e_keys_pmp<<<gridHE, BS>>>(d_F, nF, maxV, d_sort_keys,
                                             d_sort_indices);
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<long long> dp_keys(d_sort_keys);
        thrust::device_ptr<int> dp_si(d_sort_indices);
        thrust::sort_by_key(thrust::device, dp_keys, dp_keys + nHE, dp_si);
        k_pair_e2e_pmp<<<gridHE, BS>>>(d_sort_keys, d_sort_indices, nHE, d_E2E);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("[SPLIT-CUDA] %d total splits, nV=%d nF=%d\n", total_splits, nV, nF);

    // Download results
    *nV_out = nV;
    *nF_out = nF;
    *V_out = (float*)malloc(3 * nV * sizeof(float));
    *F_out = (int*)malloc(3 * nF * sizeof(int));
    *vsizing_out = (float*)malloc(nV * sizeof(float));
    *vfeature_out = (int*)malloc(nV * sizeof(int));
    *vboundary_out = (int*)malloc(nV * sizeof(int));

    CUDA_CHECK(cudaMemcpy(*V_out, d_V, 3 * nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*F_out, d_F, 3 * nF * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*vsizing_out, d_vsizing, nV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*vfeature_out, d_vfeature, nV * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*vboundary_out, d_vboundary, nV * sizeof(int), cudaMemcpyDeviceToHost));

    // Free GPU
    cudaFree(d_V); cudaFree(d_F); cudaFree(d_E2E);
    cudaFree(d_vsizing); cudaFree(d_vfeature); cudaFree(d_vboundary);
    cudaFree(d_efeature);
    cudaFree(d_marks); cudaFree(d_face_counts);
    cudaFree(d_vtx_scan); cudaFree(d_face_scan);
    cudaFree(d_F_old); cudaFree(d_sort_keys); cudaFree(d_sort_indices);
}

} // namespace pmp
