// SPDX-License-Identifier: MIT

#include "pmp/algorithms/remeshing_cuda.h"
#include "pmp/algorithms/remeshing_cuda_checkpoint.h"
#include "pmp/algorithms/remeshing.h"
#include "pmp/algorithms/normals.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>

// Forward declarations of CUDA pipelines
#ifdef PMP_HAS_CUDA
extern "C" {
bool pmp_cuda_device_available();
}
namespace pmp {
extern "C" void cuda_split_pipeline(
    float* h_V, int nV_in, int* h_F, int nF_in, float* h_vsizing,
    int* h_vfeature, int* h_vboundary, int* h_efeature, int max_passes,
    int capacity_mult, float** V_out, int* nV_out, int** F_out, int* nF_out,
    float** vsizing_out, int** vfeature_out, int** vboundary_out);
extern "C" void cuda_flip_pipeline(int* faces, int* e2e, int* vhalfedge,
                                   const int* vlocked, const int* vfeature,
                                   const int* vboundary, const int* efeature,
                                   int nV, int nF, int max_passes,
                                   int capacity_mult);
extern "C" void cuda_smooth_pipeline(
    float* positions, float* normals, const int* faces, const int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature, int nV,
    int nF, const float* ref_positions, const float* ref_normals,
    const float* ref_sizing, const int* ref_faces, int ref_nV, int ref_nF,
    int smooth_iterations, int use_projection, int block_size,
    int capacity_mult);
// Persistent smooth API
extern "C" void cuda_smooth_init(
    const float* ref_positions, const float* ref_normals,
    const float* ref_sizing, const int* ref_faces,
    int ref_nV, int ref_nF, int capacity_mult, void** handle_out);
extern "C" void cuda_smooth_iteration(
    void* handle,
    float* positions, float* normals, const int* faces, const int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature,
    int nV, int nF, int smooth_iterations, int use_projection);
extern "C" void cuda_smooth_cleanup(void* handle);
// Fused normals+flip+smooth
extern "C" void cuda_fused_flip_smooth(
    void* handle,
    float* positions, float* normals, int* faces, int* e2e,
    const int* vhalfedge, float* vsizing, const int* vlocked,
    const int* vfeature, const int* vboundary, const int* efeature,
    int nV, int nF,
    int flip_passes, int smooth_iterations, int use_projection,
    int* out_flips);
}
#endif

namespace pmp {

// ============================================================
// Config file parser (QuadWild fscanf pattern)
// ============================================================

static RemeshStrategy parse_strategy(const char* s)
{
    if (strcmp(s, "cuda") == 0 || strcmp(s, "CUDA") == 0)
        return RemeshStrategy::CUDA;
    return RemeshStrategy::CPU;
}

static const char* strategy_str(RemeshStrategy s)
{
    return s == RemeshStrategy::CUDA ? "cuda" : "cpu";
}

RemeshingCudaParams load_remeshing_config(const std::string& path)
{
    RemeshingCudaParams p;
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp)
    {
        printf("[REMESH-CONFIG] WARNING: Cannot open %s, using defaults\n",
               path.c_str());
        return p;
    }

    char key[256], val[256];
    while (fscanf(fp, "%255s", key) == 1)
    {
        if (key[0] == '#')
        {
            int c;
            while ((c = fgetc(fp)) != '\n' && c != EOF) ;
            continue;
        }
        if (fscanf(fp, "%255s", val) != 1) break;

        if (strcmp(key, "split_strategy") == 0) p.split_strategy = parse_strategy(val);
        else if (strcmp(key, "collapse_strategy") == 0) p.collapse_strategy = parse_strategy(val);
        else if (strcmp(key, "flip_strategy") == 0) p.flip_strategy = parse_strategy(val);
        else if (strcmp(key, "smooth_strategy") == 0) p.smooth_strategy = parse_strategy(val);
        else if (strcmp(key, "outer_iterations") == 0) p.outer_iterations = (unsigned)atoi(val);
        else if (strcmp(key, "split_passes") == 0) p.split_passes = (unsigned)atoi(val);
        else if (strcmp(key, "flip_passes") == 0) p.flip_passes = (unsigned)atoi(val);
        else if (strcmp(key, "smooth_sub_iterations") == 0) p.smooth_sub_iterations = (unsigned)atoi(val);
        else if (strcmp(key, "cuda_block_size") == 0) p.cuda_block_size = atoi(val);
        else if (strcmp(key, "gpu_capacity_multiplier") == 0) p.gpu_capacity_multiplier = atoi(val);
        else if (strcmp(key, "bvh_max_leaf_size") == 0) p.bvh_max_leaf_size = atoi(val);
        else if (strcmp(key, "checkpoint_enabled") == 0) p.checkpoint_enabled = (atoi(val) != 0);
        else if (strcmp(key, "checkpoint_dir") == 0) p.checkpoint_dir = val;
        else if (strcmp(key, "checkpoint_save_all") == 0) p.checkpoint_save_all = (atoi(val) != 0);
        else if (strcmp(key, "checkpoint_save_at") == 0) p.checkpoint_save_at = val;
        else if (strcmp(key, "checkpoint_run_from") == 0) p.checkpoint_run_from = val;
        else if (strcmp(key, "checkpoint_run_to") == 0) p.checkpoint_run_to = val;
        else printf("[REMESH-CONFIG] WARNING: Unknown key '%s'\n", key);
    }

    fclose(fp);
    printf("[REMESH-CONFIG] Loaded %s\n", path.c_str());
    printf("[REMESH-CONFIG]   split=%s collapse=%s flip=%s smooth=%s\n",
           strategy_str(p.split_strategy), strategy_str(p.collapse_strategy),
           strategy_str(p.flip_strategy), strategy_str(p.smooth_strategy));
    printf("[REMESH-CONFIG]   iterations=%u split_passes=%u flip_passes=%u smooth_sub=%u\n",
           p.outer_iterations, p.split_passes, p.flip_passes, p.smooth_sub_iterations);
    return p;
}

void save_remeshing_config(const std::string& path, const RemeshingCudaParams& p)
{
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) return;
    fprintf(fp, "split_strategy %s\n", strategy_str(p.split_strategy));
    fprintf(fp, "collapse_strategy %s\n", strategy_str(p.collapse_strategy));
    fprintf(fp, "flip_strategy %s\n", strategy_str(p.flip_strategy));
    fprintf(fp, "smooth_strategy %s\n", strategy_str(p.smooth_strategy));
    fprintf(fp, "outer_iterations %u\n", p.outer_iterations);
    fprintf(fp, "split_passes %u\n", p.split_passes);
    fprintf(fp, "flip_passes %u\n", p.flip_passes);
    fprintf(fp, "smooth_sub_iterations %u\n", p.smooth_sub_iterations);
    fprintf(fp, "cuda_block_size %d\n", p.cuda_block_size);
    fprintf(fp, "gpu_capacity_multiplier %d\n", p.gpu_capacity_multiplier);
    fprintf(fp, "bvh_max_leaf_size %d\n", p.bvh_max_leaf_size);
    fprintf(fp, "checkpoint_enabled %d\n", p.checkpoint_enabled ? 1 : 0);
    if (!p.checkpoint_dir.empty()) fprintf(fp, "checkpoint_dir %s\n", p.checkpoint_dir.c_str());
    fprintf(fp, "checkpoint_save_all %d\n", p.checkpoint_save_all ? 1 : 0);
    fclose(fp);
}

// ============================================================
// Flat mesh extraction for CUDA kernels
// ============================================================

namespace {

struct FlatMeshData
{
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<int> faces;
    std::vector<int> e2e;
    std::vector<int> vhalfedge;
    std::vector<float> vsizing;
    std::vector<int> vlocked;
    std::vector<int> vfeature;
    std::vector<int> vboundary;
    std::vector<int> efeature;
    int nV = 0, nF = 0;
};

FlatMeshData extract_flat_mesh(const SurfaceMesh& mesh)
{
    FlatMeshData fm;
    fm.nV = (int)mesh.n_vertices();
    fm.nF = (int)mesh.n_faces();

    fm.positions.resize(3 * fm.nV);
    for (auto v : mesh.vertices())
    {
        int i = (int)v.idx();
        const auto& p = mesh.position(v);
        fm.positions[3 * i] = (float)p[0];
        fm.positions[3 * i + 1] = (float)p[1];
        fm.positions[3 * i + 2] = (float)p[2];
    }

    fm.normals.resize(3 * fm.nV, 0.0f);
    auto vnormal = mesh.get_vertex_property<Point>("v:normal");
    if (vnormal)
        for (auto v : mesh.vertices())
        {
            int i = (int)v.idx();
            fm.normals[3 * i] = (float)vnormal[v][0];
            fm.normals[3 * i + 1] = (float)vnormal[v][1];
            fm.normals[3 * i + 2] = (float)vnormal[v][2];
        }

    fm.faces.resize(3 * fm.nF);
    fm.e2e.resize(3 * fm.nF, -1);
    fm.efeature.resize(3 * fm.nF, 0);
    auto ef = mesh.get_edge_property<bool>("e:feature");

    for (auto f : mesh.faces())
    {
        int fi = (int)f.idx();
        auto h = mesh.halfedge(f);
        for (int j = 0; j < 3; ++j)
        {
            int he_idx = 3 * fi + j;
            fm.faces[he_idx] = (int)mesh.from_vertex(h).idx();
            auto opp = mesh.opposite_halfedge(h);
            auto opp_face = mesh.face(opp);
            if (opp_face.is_valid())
            {
                int ofi = (int)opp_face.idx();
                auto oh = mesh.halfedge(opp_face);
                for (int k = 0; k < 3; ++k)
                {
                    if (oh == opp) { fm.e2e[he_idx] = 3 * ofi + k; break; }
                    oh = mesh.next_halfedge(oh);
                }
            }
            if (ef) fm.efeature[he_idx] = ef[mesh.edge(h)] ? 1 : 0;
            h = mesh.next_halfedge(h);
        }
    }

    fm.vhalfedge.resize(fm.nV, -1);
    for (auto v : mesh.vertices())
    {
        auto h = mesh.halfedge(v);
        if (!h.is_valid()) continue;
        auto f = mesh.face(h);
        if (f.is_valid())
        {
            int fi = (int)f.idx();
            auto fh = mesh.halfedge(f);
            for (int j = 0; j < 3; ++j)
            {
                if (fh == h) { fm.vhalfedge[(int)v.idx()] = 3 * fi + j; break; }
                fh = mesh.next_halfedge(fh);
            }
        }
    }

    fm.vsizing.resize(fm.nV, 0.0f);
    auto vs = mesh.get_vertex_property<Scalar>("v:sizing");
    if (vs) for (auto v : mesh.vertices()) fm.vsizing[(int)v.idx()] = (float)vs[v];

    fm.vlocked.resize(fm.nV, 0);
    auto vl = mesh.get_vertex_property<bool>("v:locked");
    if (vl) for (auto v : mesh.vertices()) fm.vlocked[(int)v.idx()] = vl[v] ? 1 : 0;

    fm.vfeature.resize(fm.nV, 0);
    auto vf = mesh.get_vertex_property<bool>("v:feature");
    if (vf) for (auto v : mesh.vertices()) fm.vfeature[(int)v.idx()] = vf[v] ? 1 : 0;

    fm.vboundary.resize(fm.nV, 0);
    for (auto v : mesh.vertices()) fm.vboundary[(int)v.idx()] = mesh.is_boundary(v) ? 1 : 0;

    return fm;
}

void write_back_positions(SurfaceMesh& mesh, const FlatMeshData& fm)
{
    auto points = mesh.get_vertex_property<Point>("v:point");
    auto vnormal = mesh.get_vertex_property<Point>("v:normal");
    auto vsizing = mesh.get_vertex_property<Scalar>("v:sizing");
    for (auto v : mesh.vertices())
    {
        int i = (int)v.idx();
        if (points) points[v] = Point(fm.positions[3*i], fm.positions[3*i+1], fm.positions[3*i+2]);
        if (vnormal) vnormal[v] = Point(fm.normals[3*i], fm.normals[3*i+1], fm.normals[3*i+2]);
        if (vsizing && !fm.vsizing.empty()) vsizing[v] = fm.vsizing[i];
    }
}

// Extract reference mesh flat arrays
struct FlatRefMesh
{
    std::vector<float> positions, normals, sizing;
    std::vector<int> faces;
    int nV = 0, nF = 0;
};

FlatRefMesh extract_ref_mesh(const SurfaceMesh& ref)
{
    FlatRefMesh r;
    r.nV = (int)ref.n_vertices();
    r.nF = (int)ref.n_faces();
    r.positions.resize(3 * r.nV);
    r.normals.resize(3 * r.nV, 0.0f);
    r.sizing.resize(r.nV, 0.0f);
    auto rp = ref.get_vertex_property<Point>("v:point");
    auto rn = ref.get_vertex_property<Point>("v:normal");
    auto rs = ref.get_vertex_property<Scalar>("v:sizing");
    for (auto v : ref.vertices())
    {
        int i = (int)v.idx();
        r.positions[3*i] = (float)rp[v][0]; r.positions[3*i+1] = (float)rp[v][1]; r.positions[3*i+2] = (float)rp[v][2];
        if (rn) { r.normals[3*i] = (float)rn[v][0]; r.normals[3*i+1] = (float)rn[v][1]; r.normals[3*i+2] = (float)rn[v][2]; }
        if (rs) r.sizing[i] = (float)rs[v];
    }
    r.faces.resize(3 * r.nF);
    size_t fi = 0;
    for (auto f : ref.faces())
    {
        auto fv = ref.vertices(f);
        r.faces[3*fi] = (int)(*fv).idx(); ++fv;
        r.faces[3*fi+1] = (int)(*fv).idx(); ++fv;
        r.faces[3*fi+2] = (int)(*fv).idx();
        ++fi;
    }
    return r;
}

// Rebuild SurfaceMesh from flat arrays (after GPU flip changed topology)
// Preserves vertex properties by index (nV doesn't change for flip)
void rebuild_mesh_from_flat(SurfaceMesh& mesh, const FlatMeshData& fm)
{
    // Save properties before clear
    std::vector<float> saved_sizing = fm.vsizing;
    std::vector<int> saved_vlocked = fm.vlocked;
    std::vector<int> saved_vfeature = fm.vfeature;

    mesh.clear();

    // Add vertices
    for (int i = 0; i < fm.nV; ++i)
        mesh.add_vertex(Point(fm.positions[3*i], fm.positions[3*i+1], fm.positions[3*i+2]));

    // Add faces
    for (int i = 0; i < fm.nF; ++i)
    {
        auto f = mesh.add_triangle(Vertex(fm.faces[3*i]), Vertex(fm.faces[3*i+1]),
                                   Vertex(fm.faces[3*i+2]));
        if (!f.is_valid())
        {
            // If add_triangle fails, skip this face and continue
            // This handles any rare non-manifold cases from GPU flip
            continue;
        }
    }

    // Restore properties
    if (!saved_sizing.empty())
    {
        auto vs = mesh.vertex_property<Scalar>("v:sizing", Scalar(0));
        for (auto v : mesh.vertices()) vs[v] = saved_sizing[(int)v.idx()];
    }
    if (!saved_vlocked.empty())
    {
        auto vl = mesh.vertex_property<bool>("v:locked", false);
        for (auto v : mesh.vertices()) vl[v] = saved_vlocked[(int)v.idx()] != 0;
    }
    if (!saved_vfeature.empty())
    {
        auto vf = mesh.vertex_property<bool>("v:feature", false);
        for (auto v : mesh.vertices()) vf[v] = saved_vfeature[(int)v.idx()] != 0;
    }

    // Edge properties need to be re-detected since edge indexing changed
    mesh.edge_property<bool>("e:feature", false);
    mesh.edge_property<bool>("e:locked", false);

    // Normals already in fm (computed on GPU)
    auto vn = mesh.vertex_property<Point>("v:normal");
    for (auto v : mesh.vertices())
    {
        int i = (int)v.idx();
        vn[v] = Point(fm.normals[3*i], fm.normals[3*i+1], fm.normals[3*i+2]);
    }
}

} // anonymous namespace

// ============================================================
// CUDA operation wrappers
// ============================================================

#ifdef PMP_HAS_CUDA

static void cuda_do_split(SurfaceMesh& mesh, int max_passes, int cap_mult,
                           Scalar target_edge_length)
{
    auto fm = extract_flat_mesh(mesh);
    float* V_out = nullptr; int* F_out = nullptr;
    float* vs_out = nullptr; int* vf_out = nullptr; int* vb_out = nullptr;
    int nV_out = 0, nF_out = 0;

    cuda_split_pipeline(
        fm.positions.data(), fm.nV, fm.faces.data(), fm.nF,
        fm.vsizing.data(), fm.vfeature.data(), fm.vboundary.data(),
        fm.efeature.data(), max_passes, cap_mult,
        &V_out, &nV_out, &F_out, &nF_out, &vs_out, &vf_out, &vb_out);

    if (nV_out > fm.nV || nF_out > fm.nF)
    {
        mesh.clear();
        for (int i = 0; i < nV_out; ++i)
            mesh.add_vertex(Point(V_out[3*i], V_out[3*i+1], V_out[3*i+2]));
        for (int i = 0; i < nF_out; ++i)
            mesh.add_triangle(Vertex(F_out[3*i]), Vertex(F_out[3*i+1]), Vertex(F_out[3*i+2]));
        auto vs = mesh.vertex_property<Scalar>("v:sizing", target_edge_length);
        auto vf = mesh.vertex_property<bool>("v:feature", false);
        auto vl = mesh.vertex_property<bool>("v:locked", false);
        for (int i = 0; i < nV_out; ++i) { vs[Vertex(i)] = vs_out[i]; vf[Vertex(i)] = vf_out[i] != 0; }
    }
    free(V_out); free(F_out); free(vs_out); free(vf_out); free(vb_out);
}

static void cuda_do_flip(SurfaceMesh& mesh, int max_passes, int cap_mult)
{
    auto fm = extract_flat_mesh(mesh);
    std::vector<int> orig_faces = fm.faces;

    cuda_flip_pipeline(fm.faces.data(), fm.e2e.data(), fm.vhalfedge.data(),
                       fm.vlocked.data(), fm.vfeature.data(),
                       fm.vboundary.data(), fm.efeature.data(),
                       fm.nV, fm.nF, max_passes, cap_mult);

    // Apply GPU-identified flips to CPU mesh
    for (int fi = 0; fi < fm.nF; ++fi)
    {
        bool changed = false;
        for (int j = 0; j < 3; ++j)
            if (fm.faces[3*fi+j] != orig_faces[3*fi+j]) { changed = true; break; }
        if (!changed) continue;

        int ov[3] = {orig_faces[3*fi], orig_faces[3*fi+1], orig_faces[3*fi+2]};
        int nv[3] = {fm.faces[3*fi], fm.faces[3*fi+1], fm.faces[3*fi+2]};
        int shared[2]; int ns = 0;
        for (int j = 0; j < 3 && ns < 2; ++j)
            for (int k = 0; k < 3; ++k)
                if (ov[j] == nv[k]) { shared[ns++] = ov[j]; break; }
        if (ns != 2) continue;

        auto h = mesh.find_halfedge(Vertex(shared[0]), Vertex(shared[1]));
        if (h.is_valid())
        {
            auto e = mesh.edge(h);
            if (!mesh.is_boundary(e) && mesh.is_flip_ok(e))
                mesh.flip(e);
        }
    }
}

static void cuda_do_smooth(SurfaceMesh& mesh, unsigned int iterations,
                            bool use_projection, const SurfaceMesh* refmesh,
                            int block_size, int cap_mult)
{
    auto fm = extract_flat_mesh(mesh);

    FlatRefMesh ref;
    if (use_projection && refmesh)
        ref = extract_ref_mesh(*refmesh);

    cuda_smooth_pipeline(
        fm.positions.data(), fm.normals.data(), fm.faces.data(), fm.e2e.data(),
        fm.vhalfedge.data(), fm.vsizing.data(), fm.vlocked.data(),
        fm.vfeature.data(), fm.vboundary.data(), fm.efeature.data(),
        fm.nV, fm.nF,
        use_projection ? ref.positions.data() : nullptr,
        use_projection ? ref.normals.data() : nullptr,
        use_projection ? ref.sizing.data() : nullptr,
        use_projection ? ref.faces.data() : nullptr,
        ref.nV, ref.nF,
        (int)iterations, use_projection ? 1 : 0, block_size, cap_mult);

    write_back_positions(mesh, fm);
}

#endif

// ============================================================
// Checkpoint helpers
// ============================================================

namespace {

RemeshCheckpointData extract_checkpoint_data(const SurfaceMesh& mesh)
{
    RemeshCheckpointData data;
    size_t nv = mesh.n_vertices(), nf = mesh.n_faces();
    data.positions.resize(3 * nv);
    size_t vi = 0;
    for (auto v : mesh.vertices())
    {
        const auto& p = mesh.position(v);
        data.positions[3*vi] = (float)p[0]; data.positions[3*vi+1] = (float)p[1]; data.positions[3*vi+2] = (float)p[2];
        ++vi;
    }
    data.faces.resize(3 * nf);
    size_t fi = 0;
    for (auto f : mesh.faces())
    {
        auto fv = mesh.vertices(f);
        data.faces[3*fi] = (int32_t)(*fv).idx(); ++fv;
        data.faces[3*fi+1] = (int32_t)(*fv).idx(); ++fv;
        data.faces[3*fi+2] = (int32_t)(*fv).idx(); ++fi;
    }
    return data;
}

} // anonymous namespace

// ============================================================
// Decomposed pipeline orchestrator
// ============================================================

void uniform_remeshing_cuda(SurfaceMesh& mesh, Scalar edge_length,
                            const RemeshingCudaParams& params,
                            bool use_projection)
{
    auto t_total = std::chrono::high_resolution_clock::now();
    RemeshingCudaParams p = params;

    printf("[REMESH] uniform_remeshing_cuda: target_edge=%.4f, iterations=%u\n",
           (double)edge_length, p.outer_iterations);
    printf("[REMESH]   strategies: split=%s collapse=%s flip=%s smooth=%s\n",
           strategy_str(p.split_strategy), strategy_str(p.collapse_strategy),
           strategy_str(p.flip_strategy), strategy_str(p.smooth_strategy));

    // Check GPU availability and fallback
    bool has_gpu = false;
#ifdef PMP_HAS_CUDA
    has_gpu = pmp_cuda_device_available();
#endif
    if (!has_gpu)
    {
        if (p.split_strategy == RemeshStrategy::CUDA) p.split_strategy = RemeshStrategy::CPU;
        if (p.flip_strategy == RemeshStrategy::CUDA) p.flip_strategy = RemeshStrategy::CPU;
        if (p.smooth_strategy == RemeshStrategy::CUDA) p.smooth_strategy = RemeshStrategy::CPU;
        if (p.collapse_strategy == RemeshStrategy::CUDA) p.collapse_strategy = RemeshStrategy::CPU;
    }
    if (p.collapse_strategy == RemeshStrategy::CUDA)
    {
        printf("[REMESH] CUDA collapse not implemented, using CPU\n");
        p.collapse_strategy = RemeshStrategy::CPU;
    }

    bool any_cuda = (p.split_strategy == RemeshStrategy::CUDA ||
                     p.flip_strategy == RemeshStrategy::CUDA ||
                     p.smooth_strategy == RemeshStrategy::CUDA);

    // If all CPU, delegate to original implementation
    if (!any_cuda)
    {
        pmp::uniform_remeshing(mesh, edge_length, p.outer_iterations, use_projection);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("[REMESH] Done: %zu verts, %zu faces (%.3f s)\n",
               mesh.n_vertices(), mesh.n_faces(),
               std::chrono::duration<double>(t1 - t_total).count());
        return;
    }

    // ============================================================
    // DECOMPOSED LOOP: use Remeshing class for CPU ops,
    // CUDA kernels for GPU ops, per-iteration dispatch
    // ============================================================

    auto engine = std::make_unique<Remeshing>(mesh);
    engine->set_uniform(edge_length, use_projection);
    engine->preprocessing();

    // Build persistent GPU state (BVH + ref mesh uploaded once)
    void* gpu_handle = nullptr;
#ifdef PMP_HAS_CUDA
    if (any_cuda && use_projection)
    {
        auto refmesh = std::make_shared<SurfaceMesh>(mesh);
        vertex_normals(*refmesh);
        auto rs = refmesh->vertex_property<Scalar>("v:sizing", edge_length);
        for (auto v : refmesh->vertices()) rs[v] = edge_length;

        auto ref = extract_ref_mesh(*refmesh);
        cuda_smooth_init(ref.positions.data(), ref.normals.data(),
                         ref.sizing.data(), ref.faces.data(),
                         ref.nV, ref.nF, p.gpu_capacity_multiplier,
                         &gpu_handle);
    }
#endif

    double t_split = 0, t_collapse = 0, t_gpu = 0;

    for (unsigned int iter = 0; iter < p.outer_iterations; ++iter)
    {
        // 1. SPLIT (CPU)
        auto ts = std::chrono::high_resolution_clock::now();
        engine->split_long_edges();
        auto te = std::chrono::high_resolution_clock::now();
        t_split += std::chrono::duration<double>(te - ts).count();

        // 2. CPU NORMALS (needed before collapse)
        vertex_normals(mesh);

        // 3. COLLAPSE (CPU)
        ts = std::chrono::high_resolution_clock::now();
        engine->collapse_short_edges();
        te = std::chrono::high_resolution_clock::now();
        t_collapse += std::chrono::duration<double>(te - ts).count();

        // 4+5. FUSED GPU PHASE: normals + flip + smooth
        //      Single extract → upload → GPU normals → GPU flip → GPU smooth → download
        //      Then rebuild SurfaceMesh from flat arrays for next iteration
        ts = std::chrono::high_resolution_clock::now();
        if (any_cuda)
        {
#ifdef PMP_HAS_CUDA
            auto fm = extract_flat_mesh(mesh);
            int n_flips = 0;

            cuda_fused_flip_smooth(gpu_handle,
                fm.positions.data(), fm.normals.data(),
                fm.faces.data(), fm.e2e.data(),
                fm.vhalfedge.data(), fm.vsizing.data(),
                fm.vlocked.data(), fm.vfeature.data(),
                fm.vboundary.data(), fm.efeature.data(),
                fm.nV, fm.nF,
                (int)p.flip_passes, (int)p.smooth_sub_iterations,
                use_projection ? 1 : 0, &n_flips);

            if (n_flips > 0)
            {
                // Topology changed — full mesh rebuild from flat arrays
                rebuild_mesh_from_flat(mesh, fm);
                // Reinitialize Remeshing engine with new mesh topology
                engine = std::make_unique<Remeshing>(mesh);
                engine->set_uniform(edge_length, use_projection);
                engine->preprocessing();
            }
            else
            {
                // No topology change — just write back positions/normals/sizing
                write_back_positions(mesh, fm);
            }
#endif
        }
        else
        {
            engine->flip_edges();
            engine->tangential_smoothing(p.smooth_sub_iterations);
        }
        te = std::chrono::high_resolution_clock::now();
        t_gpu += std::chrono::duration<double>(te - ts).count();
    }

#ifdef PMP_HAS_CUDA
    if (gpu_handle) cuda_smooth_cleanup(gpu_handle);
#endif

    engine->remove_caps();
    engine->postprocessing();

    auto t1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t1 - t_total).count();
    printf("[REMESH] Done: %zu verts, %zu faces (%.3f s)\n",
           mesh.n_vertices(), mesh.n_faces(), total);
    printf("[REMESH]   split=%.3fs  collapse=%.3fs  gpu(norms+flip+smooth)=%.3fs\n",
           t_split, t_collapse, t_gpu);

    // Checkpoint
    if (p.checkpoint_enabled && !p.checkpoint_dir.empty() && p.checkpoint_save_all)
    {
        auto data = extract_checkpoint_data(mesh);
        RemeshCheckpointHeader hdr = {};
        memcpy(hdr.magic, "RMC", 4); hdr.version = 1;
        strncpy(hdr.stage, "complete", sizeof(hdr.stage) - 1);
        hdr.num_vertices = (int32_t)mesh.n_vertices();
        hdr.num_faces = (int32_t)mesh.n_faces();
        hdr.target_edge_length = (float)edge_length;
        hdr.is_uniform = 1;
        hdr.split_strategy = (int32_t)p.split_strategy;
        hdr.collapse_strategy = (int32_t)p.collapse_strategy;
        hdr.flip_strategy = (int32_t)p.flip_strategy;
        hdr.smooth_strategy = (int32_t)p.smooth_strategy;
        hdr.timestamp = (int64_t)time(nullptr);
        save_remesh_checkpoint(data, REMESH_STAGE_COMPLETE, p.checkpoint_dir.c_str(), hdr);
    }
}

void adaptive_remeshing_cuda(SurfaceMesh& mesh, Scalar min_edge_length,
                             Scalar max_edge_length, Scalar approx_error,
                             const RemeshingCudaParams& params,
                             bool use_projection)
{
    auto t_total = std::chrono::high_resolution_clock::now();
    RemeshingCudaParams p = params;

    bool has_gpu = false;
#ifdef PMP_HAS_CUDA
    has_gpu = pmp_cuda_device_available();
#endif
    if (!has_gpu)
    {
        if (p.split_strategy == RemeshStrategy::CUDA) p.split_strategy = RemeshStrategy::CPU;
        if (p.flip_strategy == RemeshStrategy::CUDA) p.flip_strategy = RemeshStrategy::CPU;
        if (p.smooth_strategy == RemeshStrategy::CUDA) p.smooth_strategy = RemeshStrategy::CPU;
    }
    if (p.collapse_strategy == RemeshStrategy::CUDA) p.collapse_strategy = RemeshStrategy::CPU;

    bool any_cuda = (p.split_strategy == RemeshStrategy::CUDA ||
                     p.flip_strategy == RemeshStrategy::CUDA ||
                     p.smooth_strategy == RemeshStrategy::CUDA);

    if (!any_cuda)
    {
        pmp::adaptive_remeshing(mesh, min_edge_length, max_edge_length,
                                approx_error, p.outer_iterations, use_projection);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("[REMESH] Done: %zu verts, %zu faces (%.3f s)\n",
               mesh.n_vertices(), mesh.n_faces(),
               std::chrono::duration<double>(t1 - t_total).count());
        return;
    }

    auto engine = std::make_unique<Remeshing>(mesh);
    engine->set_adaptive(min_edge_length, max_edge_length, approx_error, use_projection);
    engine->preprocessing();

    void* gpu_handle = nullptr;
#ifdef PMP_HAS_CUDA
    if (any_cuda && use_projection)
    {
        auto refmesh = std::make_shared<SurfaceMesh>(mesh);
        vertex_normals(*refmesh);
        auto ref = extract_ref_mesh(*refmesh);
        cuda_smooth_init(ref.positions.data(), ref.normals.data(),
                         ref.sizing.data(), ref.faces.data(),
                         ref.nV, ref.nF, p.gpu_capacity_multiplier,
                         &gpu_handle);
    }
#endif

    for (unsigned int iter = 0; iter < p.outer_iterations; ++iter)
    {
        engine->split_long_edges();
        vertex_normals(mesh);
        engine->collapse_short_edges();

        if (any_cuda)
        {
#ifdef PMP_HAS_CUDA
            auto fm = extract_flat_mesh(mesh);
            int n_flips = 0;
            cuda_fused_flip_smooth(gpu_handle,
                fm.positions.data(), fm.normals.data(),
                fm.faces.data(), fm.e2e.data(),
                fm.vhalfedge.data(), fm.vsizing.data(),
                fm.vlocked.data(), fm.vfeature.data(),
                fm.vboundary.data(), fm.efeature.data(),
                fm.nV, fm.nF,
                (int)p.flip_passes, (int)p.smooth_sub_iterations,
                use_projection ? 1 : 0, &n_flips);

            if (n_flips > 0)
            {
                rebuild_mesh_from_flat(mesh, fm);
                engine = std::make_unique<Remeshing>(mesh);
                engine->set_adaptive(min_edge_length, max_edge_length, approx_error, use_projection);
                engine->preprocessing();
            }
            else
                write_back_positions(mesh, fm);
#endif
        }
        else
        {
            engine->flip_edges();
            engine->tangential_smoothing(p.smooth_sub_iterations);
        }
    }

#ifdef PMP_HAS_CUDA
    if (gpu_handle) cuda_smooth_cleanup(gpu_handle);
#endif

    engine->remove_caps();
    engine->postprocessing();

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[REMESH] Done: %zu verts, %zu faces (%.3f s)\n",
           mesh.n_vertices(), mesh.n_faces(),
           std::chrono::duration<double>(t1 - t_total).count());
}

} // namespace pmp
