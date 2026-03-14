// SPDX-License-Identifier: MIT

// ============================================================
// CUDA kernels for edge flipping
//
// Port of Remeshing::flip_edges() from remeshing.cpp.
// Flips edges to minimize vertex valence deviation from optimal
// (6 for interior, 4 for boundary vertices).
//
// Parallel scheme:
//   1. k_compute_valences: count vertex degree from F array
//   2. k_mark_flips: per-edge, check if flip improves valence deviation
//   3. k_resolve_flip_conflicts: independent set — edges sharing a vertex
//      cannot both flip; lower half-edge index wins
//   4. k_apply_flips: rewrite F entries for 2 faces + update E2E
//   5. k_update_valences_after_flip: adjust cached valences
//
// No vertex/face count changes — topology-only rewrites.
// ============================================================

#include "gpu_trimesh.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[FLIP] CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

namespace pmp {

__device__ inline int he_next_f(int he) { return 3 * (he / 3) + (he + 1) % 3; }
__device__ inline int he_prev_f(int he) { return 3 * (he / 3) + (he + 2) % 3; }

// ============================================================
// Kernel: compute vertex valences from F array
// ============================================================

__global__ void k_compute_valences(const int* __restrict__ F, int nF, int nV,
                                   int* __restrict__ valence)
{
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nF) return;

    // Each face contributes +1 to each of its 3 vertices
    atomicAdd(&valence[F[3 * fi + 0]], 1);
    atomicAdd(&valence[F[3 * fi + 1]], 1);
    atomicAdd(&valence[F[3 * fi + 2]], 1);
}

// ============================================================
// Kernel: mark edges that should be flipped
// ============================================================

// For each half-edge he (with he < E2E[he] to avoid double-processing),
// check if flipping improves valence deviation.
// flip_marks[he] = 1 if flip is beneficial, 0 otherwise.
// Only processes he where he < E2E[he] (canonical representative of edge).

__global__ void k_mark_flips(const int* __restrict__ F,
                             const int* __restrict__ E2E,
                             const int* __restrict__ valence,
                             const int* __restrict__ vlocked,
                             const int* __restrict__ vboundary,
                             const int* __restrict__ efeature,
                             int nHE, // = 3*nF
                             int* __restrict__ flip_marks)
{
    int he = blockIdx.x * blockDim.x + threadIdx.x;
    if (he >= nHE) return;

    flip_marks[he] = 0;

    int twin = E2E[he];
    if (twin < 0) return;           // boundary edge
    if (he > twin) return;          // only process canonical direction
    if (efeature[he]) return;       // don't flip feature edges

    // Edge vertices: source and target of he
    // In our layout: he = 3*f + j
    // Edge goes from F[he] to F[next(he)]
    int v0 = F[he];
    int v1 = F[he_next_f(he)];

    // Opposite vertices
    int v2 = F[he_prev_f(he)];      // opposite in face of he
    int v3 = F[he_prev_f(twin)];    // opposite in face of twin

    // All 4 vertices must be unlocked
    if (vlocked[v0] || vlocked[v1] || vlocked[v2] || vlocked[v3]) return;

    // is_flip_ok check: v2 != v3 and new edge (v2,v3) must not already exist
    if (v2 == v3) return;

    // Check if edge (v2,v3) already exists by walking v2's one-ring
    // via the E2E/F connectivity
    {
        int h_start = -1;
        // Find a half-edge starting at v2
        // v2 = F[prev(he)], so prev(he) has source F[prev(he)]=v2
        // Actually F[prev(he)] is v2, and prev(he) goes from v2 to F[next(prev(he))]=F[he]=v0
        // So prev(he) is an outgoing half-edge from v2
        int h_walk = he_prev_f(he);
        h_start = h_walk;
        bool found_edge = false;
        int max_iter = 64; // safety limit
        do
        {
            // to_vertex of h_walk = F[next(h_walk)]
            int to_v = F[he_next_f(h_walk)];
            if (to_v == v3) { found_edge = true; break; }
            // Rotate: prev then twin
            int prev_h = he_prev_f(h_walk);
            int tw = E2E[prev_h];
            if (tw < 0) break;
            h_walk = tw;
            --max_iter;
        } while (h_walk != h_start && max_iter > 0);

        if (found_edge) return;
    }

    // Compute valence deviation before and after flip
    int val0 = valence[v0], val1 = valence[v1];
    int val2 = valence[v2], val3 = valence[v3];

    int opt0 = vboundary[v0] ? 4 : 6;
    int opt1 = vboundary[v1] ? 4 : 6;
    int opt2 = vboundary[v2] ? 4 : 6;
    int opt3 = vboundary[v3] ? 4 : 6;

    int dev_before = (val0 - opt0) * (val0 - opt0) +
                     (val1 - opt1) * (val1 - opt1) +
                     (val2 - opt2) * (val2 - opt2) +
                     (val3 - opt3) * (val3 - opt3);

    // After flip: v0,v1 lose a neighbor; v2,v3 gain one
    int dev_after = (val0 - 1 - opt0) * (val0 - 1 - opt0) +
                    (val1 - 1 - opt1) * (val1 - 1 - opt1) +
                    (val2 + 1 - opt2) * (val2 + 1 - opt2) +
                    (val3 + 1 - opt3) * (val3 + 1 - opt3);

    if (dev_before > dev_after)
    {
        flip_marks[he] = 1;
    }
}

// ============================================================
// Kernel: resolve flip conflicts (independent set)
// ============================================================

// Two marked edges conflict if they share any vertex.
// For each marked edge, check all neighboring edges (via E2E and face traversal).
// Lower half-edge index wins.

__global__ void k_resolve_flip_conflicts(const int* __restrict__ F,
                                         const int* __restrict__ E2E,
                                         int nHE,
                                         int* __restrict__ flip_marks)
{
    int he = blockIdx.x * blockDim.x + threadIdx.x;
    if (he >= nHE) return;
    if (!flip_marks[he]) return;

    int twin = E2E[he];

    // Get all 4 vertices of this edge's diamond
    int v0 = F[he];
    int v1 = F[he_next_f(he)];
    int v2 = F[he_prev_f(he)];
    int v3 = F[he_prev_f(twin)];

    // Check all other half-edges in the two incident faces
    // Face of he: 3 half-edges
    int f0 = 3 * (he / 3);
    // Face of twin: 3 half-edges
    int f1 = 3 * (twin / 3);

    // For each neighboring edge (via E2E of same-face edges), check if it's
    // also marked and has a higher index than us
    for (int j = 0; j < 3; ++j)
    {
        // Edges in face 0 (excluding he itself)
        int neighbor_he = f0 + j;
        if (neighbor_he == he) continue;
        int nb_twin = E2E[neighbor_he];
        if (nb_twin >= 0)
        {
            // The canonical representative of this neighbor edge
            int nb_canon = (neighbor_he < nb_twin) ? neighbor_he : nb_twin;
            if (nb_canon != he && flip_marks[nb_canon])
            {
                // Conflict! Lower index wins
                if (he > nb_canon)
                {
                    flip_marks[he] = 0;
                    return;
                }
            }
        }

        // Edges in face 1 (excluding twin itself)
        neighbor_he = f1 + j;
        if (neighbor_he == twin) continue;
        nb_twin = E2E[neighbor_he];
        if (nb_twin >= 0)
        {
            int nb_canon = (neighbor_he < nb_twin) ? neighbor_he : nb_twin;
            if (nb_canon != he && flip_marks[nb_canon])
            {
                if (he > nb_canon)
                {
                    flip_marks[he] = 0;
                    return;
                }
            }
        }
    }

    // Also check edges adjacent to v2 and v3 via their face neighbors
    // (edges in the twin's face that connect to v3, and edges reachable
    // from v2 and v3 through E2E of the opposite half-edges)
    // The face-local check above covers the immediate neighborhood.
    // For a conservative approach, we also check the E2E neighbors of
    // the non-shared edges to catch diamonds sharing v2 or v3.
    for (int j = 0; j < 3; ++j)
    {
        int face_he = f0 + j;
        if (face_he == he) continue;
        int opp = E2E[face_he];
        if (opp < 0) continue;
        int opp_face = 3 * (opp / 3);
        for (int k = 0; k < 3; ++k)
        {
            int nb = opp_face + k;
            if (nb == opp) continue;
            int nb_t = E2E[nb];
            if (nb_t < 0) continue;
            int nb_c = (nb < nb_t) ? nb : nb_t;
            if (nb_c != he && flip_marks[nb_c])
            {
                if (he > nb_c) { flip_marks[he] = 0; return; }
            }
        }
    }
    for (int j = 0; j < 3; ++j)
    {
        int face_he = f1 + j;
        if (face_he == twin) continue;
        int opp = E2E[face_he];
        if (opp < 0) continue;
        int opp_face = 3 * (opp / 3);
        for (int k = 0; k < 3; ++k)
        {
            int nb = opp_face + k;
            if (nb == opp) continue;
            int nb_t = E2E[nb];
            if (nb_t < 0) continue;
            int nb_c = (nb < nb_t) ? nb : nb_t;
            if (nb_c != he && flip_marks[nb_c])
            {
                if (he > nb_c) { flip_marks[he] = 0; return; }
            }
        }
    }
}

// ============================================================
// Kernel: apply flips
// ============================================================

// For each marked edge (canonical he, with he < E2E[he]):
//
// Before:
//   Face A (he's face): vertices [v0, v1, v2] with he going v0→v1
//   Face B (twin's face): vertices [v1, v0, v3] with twin going v1→v0
//
//   he     = 3*fA + jA, goes from F[he] to F[next(he)]
//   twin   = 3*fB + jB, goes from F[twin] to F[next(twin)]
//
//   v0 = F[he], v1 = F[next(he)], v2 = F[prev(he)]
//   v3 = F[prev(twin)]
//
// After flip: edge connects v2-v3 instead of v0-v1
//   Face A becomes: [v2, v3, v0]  (he now goes v2→v3)
//   Face B becomes: [v3, v2, v1]  (twin now goes v3→v2)

__global__ void k_apply_flips(int* __restrict__ F, int* __restrict__ E2E,
                              int* __restrict__ vhalfedge,
                              const int* __restrict__ flip_marks, int nHE)
{
    int he = blockIdx.x * blockDim.x + threadIdx.x;
    if (he >= nHE) return;
    if (!flip_marks[he]) return;

    int twin = E2E[he];

    // Current vertices
    int jA = he % 3;
    int fA = he / 3;
    int jB = twin % 3;
    int fB = twin / 3;

    int v0 = F[3 * fA + jA];               // source of he
    int v1 = F[3 * fA + (jA + 1) % 3];     // target of he
    int v2 = F[3 * fA + (jA + 2) % 3];     // opposite in face A
    int v3 = F[3 * fB + (jB + 2) % 3];     // opposite in face B

    // E2E of the 4 non-shared half-edges (the edges that connect to v2 and v3)
    int heA1 = 3 * fA + (jA + 1) % 3;  // v1→v2 in old face A
    int heA2 = 3 * fA + (jA + 2) % 3;  // v2→v0 in old face A
    int heB1 = 3 * fB + (jB + 1) % 3;  // v0→v3 in old face B
    int heB2 = 3 * fB + (jB + 2) % 3;  // v3→v1 in old face B

    int twinA1 = E2E[heA1];
    int twinA2 = E2E[heA2];
    int twinB1 = E2E[heB1];
    int twinB2 = E2E[heB2];

    // New face A: v0, v2, v3  (he goes from some vertex arrangement)
    // Actually, let's keep the half-edge at position jA as the flipped edge.
    // New arrangement:
    //   Face A: he (pos jA) = v2→v3, (jA+1) = v3→v0, (jA+2) = v0→v2
    //   Face B: twin (pos jB) = v3→v2, (jB+1) = v2→v1, (jB+2) = v1→v3
    //
    // This means:
    //   F[3*fA + jA] = v2
    //   F[3*fA + (jA+1)%3] = v3
    //   F[3*fA + (jA+2)%3] = v0
    //   F[3*fB + jB] = v3
    //   F[3*fB + (jB+1)%3] = v2
    //   F[3*fB + (jB+2)%3] = v1

    F[3 * fA + jA] = v2;
    F[3 * fA + (jA + 1) % 3] = v3;
    F[3 * fA + (jA + 2) % 3] = v0;
    F[3 * fB + jB] = v3;
    F[3 * fB + (jB + 1) % 3] = v2;
    F[3 * fB + (jB + 2) % 3] = v1;

    // Update E2E:
    // The flipped edge: he ↔ twin (unchanged pairing)
    // E2E[he] = twin, E2E[twin] = he (already correct)

    // The 4 non-shared edges need new E2E mappings:
    // New face A at (jA+1): v3→v0, was old face B at (jB+1): v0→v3
    //   This half-edge's twin should be twinB1 (the outside neighbor)
    // New face A at (jA+2): v0→v2, was old face A at (jA+2): v2→v0
    //   This half-edge's twin should be twinA2

    // New face B at (jB+1): v2→v1, was old face A at (jA+1): v1→v2
    //   This half-edge's twin should be twinA1
    // New face B at (jB+2): v1→v3, was old face B at (jB+2): v3→v1
    //   This half-edge's twin should be twinB2

    // Positions in the new faces:
    int newA1 = 3 * fA + (jA + 1) % 3; // v3→v0
    int newA2 = 3 * fA + (jA + 2) % 3; // v0→v2
    int newB1 = 3 * fB + (jB + 1) % 3; // v2→v1
    int newB2 = 3 * fB + (jB + 2) % 3; // v1→v3

    E2E[newA1] = twinB1;
    if (twinB1 >= 0) E2E[twinB1] = newA1;

    E2E[newA2] = twinA2;
    if (twinA2 >= 0) E2E[twinA2] = newA2;

    E2E[newB1] = twinA1;
    if (twinA1 >= 0) E2E[twinA1] = newB1;

    E2E[newB2] = twinB2;
    if (twinB2 >= 0) E2E[twinB2] = newB2;

    // Update vhalfedge for v0 and v1 (they may have lost their outgoing edge)
    // v0's outgoing halfedge: newA2 (v0→v2) is a valid outgoing from v0
    vhalfedge[v0] = newA2;
    // v1's outgoing halfedge: newB2 (v1→v3) is a valid outgoing from v1
    vhalfedge[v1] = newB2;
    // v2: he (v2→v3) is valid
    vhalfedge[v2] = he;
    // v3: twin (v3→v2) is valid
    vhalfedge[v3] = twin;
}

// ============================================================
// Kernel: update valences after flips
// ============================================================

__global__ void k_update_valences_after_flip(const int* __restrict__ F,
                                             const int* __restrict__ E2E,
                                             const int* __restrict__ flip_marks,
                                             int nHE,
                                             int* __restrict__ valence)
{
    int he = blockIdx.x * blockDim.x + threadIdx.x;
    if (he >= nHE) return;
    if (!flip_marks[he]) return;

    // After applying the flip, the new F has the updated vertices.
    // The old v0, v1 lost one neighbor; old v2, v3 gained one.
    // But we've already rewritten F, so we need the original vertices.
    // Instead, we pass this info through: the flip mark was set on canonical he,
    // and after k_apply_flips, the new face A has F[he]=v2, F[next]=v3, F[prev]=v0,
    // and new face B has F[twin]=v3, F[next]=v2, F[prev]=v1.

    int twin = E2E[he];
    int jA = he % 3;
    int fA = he / 3;
    int jB = twin % 3;
    int fB = twin / 3;

    // After flip: v2 = F[3*fA + jA], v3 = F[3*fA + (jA+1)%3],
    //             v0 = F[3*fA + (jA+2)%3], v1 = F[3*fB + (jB+2)%3]
    int v0 = F[3 * fA + (jA + 2) % 3];
    int v1 = F[3 * fB + (jB + 2) % 3];
    int v2 = F[3 * fA + jA];
    int v3 = F[3 * fA + (jA + 1) % 3];

    atomicAdd(&valence[v0], -1);
    atomicAdd(&valence[v1], -1);
    atomicAdd(&valence[v2], 1);
    atomicAdd(&valence[v3], 1);
}

// ============================================================
// Host entry: run flip passes on GPU
// ============================================================

// Returns number of flips performed
int cuda_flip_edges(GpuTriMesh& gm, int max_passes)
{
    const int BS = 256;
    int nHE = 3 * gm.nF;
    int gridHE = (nHE + BS - 1) / BS;
    int gridF = (gm.nF + BS - 1) / BS;

    // Allocate temporaries
    int* d_valence;
    int* d_flip_marks;
    CUDA_CHECK(cudaMalloc(&d_valence, gm.nV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flip_marks, nHE * sizeof(int)));

    // Compute initial valences
    CUDA_CHECK(cudaMemset(d_valence, 0, gm.nV * sizeof(int)));
    k_compute_valences<<<gridF, BS>>>(gm.d_F, gm.nF, gm.nV, d_valence);
    CUDA_CHECK(cudaDeviceSynchronize());

    int total_flips = 0;

    for (int pass = 0; pass < max_passes; ++pass)
    {
        // Mark beneficial flips
        CUDA_CHECK(cudaMemset(d_flip_marks, 0, nHE * sizeof(int)));
        k_mark_flips<<<gridHE, BS>>>(gm.d_F, gm.d_E2E, d_valence,
                                     gm.d_vlocked, gm.d_vboundary,
                                     gm.d_efeature, nHE, d_flip_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Resolve conflicts
        k_resolve_flip_conflicts<<<gridHE, BS>>>(gm.d_F, gm.d_E2E, nHE,
                                                 d_flip_marks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Count flips this pass
        // (We could use thrust::reduce but a simple download is fine for now)
        std::vector<int> h_marks(nHE);
        CUDA_CHECK(cudaMemcpy(h_marks.data(), d_flip_marks, nHE * sizeof(int),
                              cudaMemcpyDeviceToHost));
        int n_flips = 0;
        for (int i = 0; i < nHE; ++i)
            n_flips += h_marks[i];

        if (n_flips == 0) break;

        // Apply flips
        k_apply_flips<<<gridHE, BS>>>(gm.d_F, gm.d_E2E, gm.d_vhalfedge,
                                      d_flip_marks, nHE);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update valences
        k_update_valences_after_flip<<<gridHE, BS>>>(
            gm.d_F, gm.d_E2E, d_flip_marks, nHE, d_valence);
        CUDA_CHECK(cudaDeviceSynchronize());

        total_flips += n_flips;
    }

    cudaFree(d_valence);
    cudaFree(d_flip_marks);

    return total_flips;
}

// ============================================================
// Host entry: full flip pipeline (flat arrays in/out)
// ============================================================

extern "C" void cuda_flip_pipeline(int* faces, int* e2e, int* vhalfedge,
                                   const int* vlocked, const int* vfeature,
                                   const int* vboundary, const int* efeature,
                                   int nV, int nF, int max_passes,
                                   int capacity_mult)
{
    GpuTriMesh gm;
    gpu_trimesh_alloc(gm, nV * capacity_mult, nF * capacity_mult);
    gm.nV = nV;
    gm.nF = nF;

    int nHE = 3 * nF;

    CUDA_CHECK(cudaMemcpy(gm.d_F, faces, nHE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_E2E, e2e, nHE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vhalfedge, vhalfedge, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vlocked, vlocked, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vfeature, vfeature, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_vboundary, vboundary, nV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gm.d_efeature, efeature, nHE * sizeof(int), cudaMemcpyHostToDevice));

    // Don't need V, vnormal, vsizing, update for flipping
    // (V is needed to verify is_flip_ok with find_halfedge, but we skip that check)

    int total = cuda_flip_edges(gm, max_passes);
    printf("[FLIP-CUDA] %d flips in up to %d passes\n", total, max_passes);

    // Download modified arrays
    CUDA_CHECK(cudaMemcpy(faces, gm.d_F, nHE * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e2e, gm.d_E2E, nHE * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vhalfedge, gm.d_vhalfedge, nV * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_trimesh_free(gm);
}

} // namespace pmp
