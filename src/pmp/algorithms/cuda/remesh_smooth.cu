// SPDX-License-Identifier: MIT

// ============================================================
// CUDA kernels for tangential smoothing
//
// Port of Remeshing::tangential_smoothing() from remeshing.cpp.
// Each vertex independently: walk one-ring via E2E, compute update
// (minimize_squared_areas or weighted_centroid), subtract normal
// component, then apply Jacobi-style.
// ============================================================

#include "gpu_trimesh.cuh"
#include "gpu_bvh.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[SMOOTH] CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

namespace pmp {

// ============================================================
// Device: one-ring traversal helpers
// ============================================================

// In a triangle mesh with F[3*nF] layout:
//   halfedge he = 3*face + local_edge (0,1,2)
//   next(he) = 3*(he/3) + (he+1)%3
//   prev(he) = 3*(he/3) + (he+2)%3
//   twin(he) = E2E[he]
//   to_vertex(he) = F[he]  (vertex at tip of halfedge)
//
// vhalfedge[v] = an outgoing halfedge from v

__device__ inline int he_next(int he) { return 3 * (he / 3) + (he + 1) % 3; }
__device__ inline int he_prev(int he) { return 3 * (he / 3) + (he + 2) % 3; }

// ============================================================
// Kernel: compute area-weighted vertex normals
// ============================================================

__global__ void k_compute_vertex_normals(const float* __restrict__ V,
                                         const int* __restrict__ F, int nF,
                                         int nV,
                                         float* __restrict__ vnormal)
{
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nF) return;

    int i0 = F[3 * fi], i1 = F[3 * fi + 1], i2 = F[3 * fi + 2];
    float v0x = V[3 * i0], v0y = V[3 * i0 + 1], v0z = V[3 * i0 + 2];
    float v1x = V[3 * i1], v1y = V[3 * i1 + 1], v1z = V[3 * i1 + 2];
    float v2x = V[3 * i2], v2y = V[3 * i2 + 1], v2z = V[3 * i2 + 2];

    // Face normal (not normalized — area-weighted)
    float ex = v1x - v0x, ey = v1y - v0y, ez = v1z - v0z;
    float fx = v2x - v0x, fy = v2y - v0y, fz = v2z - v0z;
    float nx = ey * fz - ez * fy;
    float ny = ez * fx - ex * fz;
    float nz = ex * fy - ey * fx;

    // Scatter to all 3 vertices
    atomicAdd(&vnormal[3 * i0], nx);
    atomicAdd(&vnormal[3 * i0 + 1], ny);
    atomicAdd(&vnormal[3 * i0 + 2], nz);
    atomicAdd(&vnormal[3 * i1], nx);
    atomicAdd(&vnormal[3 * i1 + 1], ny);
    atomicAdd(&vnormal[3 * i1 + 2], nz);
    atomicAdd(&vnormal[3 * i2], nx);
    atomicAdd(&vnormal[3 * i2 + 1], ny);
    atomicAdd(&vnormal[3 * i2 + 2], nz);
}

__global__ void k_normalize_normals(float* __restrict__ vnormal, int nV)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nV) return;

    float nx = vnormal[3 * vi], ny = vnormal[3 * vi + 1],
          nz = vnormal[3 * vi + 2];
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-10f)
    {
        float inv = 1.0f / len;
        vnormal[3 * vi] = nx * inv;
        vnormal[3 * vi + 1] = ny * inv;
        vnormal[3 * vi + 2] = nz * inv;
    }
}

// ============================================================
// Kernel: compute smooth update per vertex
// ============================================================

__global__ void k_compute_smooth_update(
    const float* __restrict__ V,
    const int* __restrict__ F,
    const int* __restrict__ E2E,
    const int* __restrict__ vhalfedge,
    const float* __restrict__ vnormal,
    const float* __restrict__ vsizing,
    const int* __restrict__ vlocked,
    const int* __restrict__ vfeature,
    const int* __restrict__ vboundary,
    const int* __restrict__ efeature,
    int nV,
    float* __restrict__ update)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nV) return;

    update[3 * vi] = update[3 * vi + 1] = update[3 * vi + 2] = 0.0f;

    if (vboundary[vi] || vlocked[vi]) return;

    float px = V[3 * vi], py = V[3 * vi + 1], pz = V[3 * vi + 2];

    if (vfeature[vi])
    {
        // Feature vertex: move along feature edges only
        float ux = 0, uy = 0, uz = 0;
        float tx = 0, ty = 0, tz = 0;
        float ww = 0;
        int c = 0;

        int h_start = vhalfedge[vi];
        if (h_start < 0) return;
        int h = h_start;
        do
        {
            // The target vertex of halfedge h is F[h]
            // But we need the outgoing halfedge from vi
            // vhalfedge stores an outgoing halfedge: v -> to_vertex
            // to_vertex(h) = F[he_next(h)] actually... let me think.
            // In our layout: for halfedge he in face f, F[he] is the vertex
            // at the START of the next edge. Actually in PMP convention,
            // F[3*f+j] is vertex j of face f.
            // For halfedge he = 3*f+j, the edge goes from F[3*f+j] to F[3*f+(j+1)%3]
            // So to_vertex(he) = F[he_next(he)]
            // vhalfedge[v] is an outgoing he from v, meaning F[he] == v? No.
            // Actually: if he is an outgoing halfedge from v, then the edge goes
            // from v to to_vertex. In our representation, he = 3*f+j means the
            // edge from F[3*f+j] to F[3*f+(j+1)%3]. So F[he] is the source vertex.
            // to_vertex(he) = F[next(he)].

            int to_v = F[he_next(h)];

            // Check if this edge is a feature edge
            // Edge index: he / 2 in PMP, but here we use half-edge feature flags
            int is_feat = efeature[h];

            if (is_feat)
            {
                float tvx = V[3 * to_v], tvy = V[3 * to_v + 1],
                      tvz = V[3 * to_v + 2];

                // Midpoint
                float bx = 0.5f * (px + tvx), by = 0.5f * (py + tvy),
                      bz = 0.5f * (pz + tvz);

                float dx = tvx - px, dy = tvy - py, dz = tvz - pz;
                float dist =
                    sqrtf(dx * dx + dy * dy + dz * dz);
                float avg_s = 0.5f * (vsizing[vi] + vsizing[to_v]);
                float w = (avg_s > 1e-10f) ? dist / avg_s : 1.0f;

                ww += w;
                ux += w * bx;
                uy += w * by;
                uz += w * bz;

                float len = dist;
                if (len > 1e-10f)
                {
                    float inv = 1.0f / len;
                    if (c == 0)
                    {
                        tx += dx * inv;
                        ty += dy * inv;
                        tz += dz * inv;
                    }
                    else
                    {
                        tx -= dx * inv;
                        ty -= dy * inv;
                        tz -= dz * inv;
                    }
                }
                ++c;
            }

            // Rotate: prev then twin
            int prev_h = he_prev(h);
            h = E2E[prev_h];
            if (h < 0) break; // boundary
        } while (h != h_start);

        if (c == 2 && ww > 1e-10f)
        {
            ux = ux / ww - px;
            uy = uy / ww - py;
            uz = uz / ww - pz;

            float tlen = sqrtf(tx * tx + ty * ty + tz * tz);
            if (tlen > 1e-10f)
            {
                tx /= tlen;
                ty /= tlen;
                tz /= tlen;
            }
            float d = ux * tx + uy * ty + uz * tz;
            update[3 * vi] = tx * d;
            update[3 * vi + 1] = ty * d;
            update[3 * vi + 2] = tz * d;
        }
    }
    else
    {
        // Non-feature vertex: minimize_squared_areas or weighted_centroid
        // Try minimize_squared_areas first (3x3 matrix solve)
        float A00 = 0, A01 = 0, A02 = 0, A11 = 0, A12 = 0, A22 = 0;
        float bx = 0, by = 0, bz = 0;
        bool valid = true;

        int h_start = vhalfedge[vi];
        if (h_start < 0) return;
        int h = h_start;

        // Also compute weighted_centroid as fallback
        float cx = 0, cy = 0, cz = 0;
        float cww = 0;

        do
        {
            // Edge opposite to vi in this triangle:
            // vi is at F[h] (source of halfedge h)
            // The opposite edge goes from F[next(h)] to F[prev(h)]
            int v0_idx = F[he_next(h)];
            int v1_idx = F[he_prev(h)];

            float p0x = V[3 * v0_idx], p0y = V[3 * v0_idx + 1],
                  p0z = V[3 * v0_idx + 2];
            float p1x = V[3 * v1_idx], p1y = V[3 * v1_idx + 1],
                  p1z = V[3 * v1_idx + 2];

            float dx = p1x - p0x, dy = p1y - p0y, dz = p1z - p0z;
            float edge_len = sqrtf(dx * dx + dy * dy + dz * dz);
            float w = (edge_len > 1e-10f) ? 1.0f / edge_len : 1.0f;

            // D matrix: cross-product-with-d squared
            float D00 = dy * dy + dz * dz;
            float D11 = dx * dx + dz * dz;
            float D22 = dx * dx + dy * dy;
            float D01 = -dx * dy;
            float D02 = -dx * dz;
            float D12 = -dy * dz;

            A00 += w * D00; A01 += w * D01; A02 += w * D02;
            A11 += w * D11; A12 += w * D12; A22 += w * D22;

            // b += w * D * p0
            bx += w * (D00 * p0x + D01 * p0y + D02 * p0z);
            by += w * (D01 * p0x + D11 * p0y + D12 * p0z);
            bz += w * (D02 * p0x + D12 * p0y + D22 * p0z);

            // Weighted centroid fallback
            float tri_cx = (px + p0x + p1x) / 3.0f;
            float tri_cy = (py + p0y + p1y) / 3.0f;
            float tri_cz = (pz + p0z + p1z) / 3.0f;

            float ex2 = p0x - px, ey2 = p0y - py, ez2 = p0z - pz;
            float fx2 = p1x - px, fy2 = p1y - py, fz2 = p1z - pz;
            float cross_x = ey2 * fz2 - ez2 * fy2;
            float cross_y = ez2 * fx2 - ex2 * fz2;
            float cross_z = ex2 * fy2 - ey2 * fx2;
            float area = sqrtf(cross_x * cross_x + cross_y * cross_y +
                               cross_z * cross_z);
            if (area == 0.0f) area = 1.0f;
            float avg_s =
                (vsizing[vi] + vsizing[v0_idx] + vsizing[v1_idx]) / 3.0f;
            float cw = area / (avg_s * avg_s + 1e-10f);
            cx += cw * tri_cx;
            cy += cw * tri_cy;
            cz += cw * tri_cz;
            cww += cw;

            // Rotate: prev then twin
            int prev_h = he_prev(h);
            h = E2E[prev_h];
            if (h < 0) { valid = false; break; }
        } while (h != h_start);

        float new_px, new_py, new_pz;

        if (valid)
        {
            // Solve A * x = b via Cramer's rule (3x3)
            float det = A00 * (A11 * A22 - A12 * A12) -
                        A01 * (A01 * A22 - A12 * A02) +
                        A02 * (A01 * A12 - A11 * A02);

            if (fabsf(det) > 1e-10f)
            {
                float inv_det = 1.0f / det;
                // Adjugate matrix * b
                new_px = ((A11 * A22 - A12 * A12) * bx +
                          (A02 * A12 - A01 * A22) * by +
                          (A01 * A12 - A02 * A11) * bz) *
                         inv_det;
                new_py = ((A12 * A02 - A01 * A22) * bx +
                          (A00 * A22 - A02 * A02) * by +
                          (A01 * A02 - A00 * A12) * bz) *
                         inv_det;
                new_pz = ((A01 * A12 - A02 * A11) * bx +
                          (A02 * A01 - A00 * A12) * by +
                          (A00 * A11 - A01 * A01) * bz) *
                         inv_det;
            }
            else
            {
                // Fallback to centroid
                if (cww > 1e-10f)
                {
                    new_px = cx / cww;
                    new_py = cy / cww;
                    new_pz = cz / cww;
                }
                else
                {
                    new_px = px;
                    new_py = py;
                    new_pz = pz;
                }
            }
        }
        else
        {
            // Boundary encountered during traversal, use centroid
            if (cww > 1e-10f)
            {
                new_px = cx / cww;
                new_py = cy / cww;
                new_pz = cz / cww;
            }
            else
            {
                new_px = px;
                new_py = py;
                new_pz = pz;
            }
        }

        // Compute update and remove normal component
        float ux = new_px - px, uy = new_py - py, uz = new_pz - pz;
        float nx = vnormal[3 * vi], ny = vnormal[3 * vi + 1],
              nz = vnormal[3 * vi + 2];
        float dn = ux * nx + uy * ny + uz * nz;
        ux -= dn * nx;
        uy -= dn * ny;
        uz -= dn * nz;

        update[3 * vi] = ux;
        update[3 * vi + 1] = uy;
        update[3 * vi + 2] = uz;
    }
}

// ============================================================
// Kernel: apply smooth update
// ============================================================

__global__ void k_apply_smooth_update(float* __restrict__ V,
                                      const float* __restrict__ upd,
                                      const int* __restrict__ vlocked,
                                      const int* __restrict__ vboundary,
                                      int nV)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nV) return;

    if (!vboundary[vi] && !vlocked[vi])
    {
        V[3 * vi] += upd[3 * vi];
        V[3 * vi + 1] += upd[3 * vi + 1];
        V[3 * vi + 2] += upd[3 * vi + 2];
    }
}

// ============================================================
// Kernel: project_to_reference via BVH
// ============================================================

__global__ void k_project_to_reference(
    float* __restrict__ V,
    float* __restrict__ vnormal,
    float* __restrict__ vsizing,
    const int* __restrict__ vlocked,
    const int* __restrict__ vboundary,
    int nV,
    const NearestResult* __restrict__ bvh_results,
    const float* __restrict__ ref_vnormal,  // per-vertex normals of ref mesh [3*ref_nV]
    const float* __restrict__ ref_vsizing,  // per-vertex sizing of ref mesh [ref_nV]
    const int* __restrict__ ref_F,          // ref mesh faces [3*ref_nF]
    int ref_nV)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nV) return;
    if (vboundary[vi] || vlocked[vi]) return;

    const NearestResult& r = bvh_results[vi];
    if (r.face_idx < 0) return;

    // Set position to nearest point
    V[3 * vi] = r.nearest_x;
    V[3 * vi + 1] = r.nearest_y;
    V[3 * vi + 2] = r.nearest_z;

    // Interpolate normal and sizing from reference mesh using barycentric coords
    int fi = r.face_idx;
    int rv0 = ref_F[3 * fi], rv1 = ref_F[3 * fi + 1], rv2 = ref_F[3 * fi + 2];

    float b0 = r.bary_u, b1 = r.bary_v, b2 = r.bary_w;

    // Interpolate normal
    float nx = b0 * ref_vnormal[3 * rv0] + b1 * ref_vnormal[3 * rv1] +
               b2 * ref_vnormal[3 * rv2];
    float ny = b0 * ref_vnormal[3 * rv0 + 1] + b1 * ref_vnormal[3 * rv1 + 1] +
               b2 * ref_vnormal[3 * rv2 + 1];
    float nz = b0 * ref_vnormal[3 * rv0 + 2] + b1 * ref_vnormal[3 * rv1 + 2] +
               b2 * ref_vnormal[3 * rv2 + 2];
    float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nlen > 1e-10f)
    {
        nx /= nlen; ny /= nlen; nz /= nlen;
    }
    vnormal[3 * vi] = nx;
    vnormal[3 * vi + 1] = ny;
    vnormal[3 * vi + 2] = nz;

    // Interpolate sizing
    vsizing[vi] = b0 * ref_vsizing[rv0] + b1 * ref_vsizing[rv1] +
                  b2 * ref_vsizing[rv2];
}

// ============================================================
// Host entry points
// ============================================================

void cuda_compute_vertex_normals(GpuTriMesh& gm)
{
    const int BS = 256;
    // Zero normals
    CUDA_CHECK(cudaMemset(gm.d_vnormal, 0, 3 * gm.nV * sizeof(float)));

    // Accumulate face normals
    int gridF = (gm.nF + BS - 1) / BS;
    k_compute_vertex_normals<<<gridF, BS>>>(gm.d_V, gm.d_F, gm.nF, gm.nV,
                                            gm.d_vnormal);

    // Normalize
    int gridV = (gm.nV + BS - 1) / BS;
    k_normalize_normals<<<gridV, BS>>>(gm.d_vnormal, gm.nV);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_tangential_smooth_iteration(GpuTriMesh& gm)
{
    const int BS = 256;
    int gridV = (gm.nV + BS - 1) / BS;

    // Compute smooth update
    k_compute_smooth_update<<<gridV, BS>>>(
        gm.d_V, gm.d_F, gm.d_E2E, gm.d_vhalfedge, gm.d_vnormal,
        gm.d_vsizing, gm.d_vlocked, gm.d_vfeature, gm.d_vboundary,
        gm.d_efeature, gm.nV, gm.d_update);

    // Apply update
    k_apply_smooth_update<<<gridV, BS>>>(gm.d_V, gm.d_update, gm.d_vlocked,
                                         gm.d_vboundary, gm.nV);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update normals
    cuda_compute_vertex_normals(gm);
}

void cuda_project_to_reference(GpuTriMesh& gm, const GpuBVH& bvh,
                               const float* d_ref_vnormal,
                               const float* d_ref_vsizing,
                               const int* d_ref_F, int ref_nV)
{
    if (gm.nV == 0) return;
    const int BS = 256;
    int gridV = (gm.nV + BS - 1) / BS;

    // Run BVH queries for all vertices
    NearestResult* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, gm.nV * sizeof(NearestResult)));
    gpu_bvh_nearest(bvh, gm.d_V, gm.nV, d_results);

    // Project
    k_project_to_reference<<<gridV, BS>>>(
        gm.d_V, gm.d_vnormal, gm.d_vsizing, gm.d_vlocked, gm.d_vboundary,
        gm.nV, d_results, d_ref_vnormal, d_ref_vsizing, d_ref_F, ref_nV);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_results);
}

} // namespace pmp
