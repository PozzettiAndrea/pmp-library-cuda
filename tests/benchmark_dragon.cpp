// Benchmark: CPU vs CUDA remeshing on Stanford Dragon
// Tests multiple target edge lengths for realistic workloads

#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing_cuda.h"
#include "pmp/algorithms/remeshing.h"
#include "pmp/algorithms/utilities.h"
#include "pmp/bounding_box.h"
#include "pmp/io/io.h"

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <numeric>

using namespace pmp;

struct Result {
    double times[2];
    size_t verts, faces;
};

static Result run(const SurfaceMesh& input, Scalar target,
                  const RemeshingCudaParams* params)
{
    Result r{{}, 0, 0};
    for (int i = 0; i < 2; ++i)
    {
        SurfaceMesh mesh = input;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (params)
            uniform_remeshing_cuda(mesh, target, *params);
        else
            uniform_remeshing(mesh, target, 10);
        auto t1 = std::chrono::high_resolution_clock::now();
        r.times[i] = std::chrono::duration<double>(t1 - t0).count();
        r.verts = mesh.n_vertices();
        r.faces = mesh.n_faces();
    }
    return r;
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1]
        : "../../../cudageom/QuadriFlow-cuda/examples/dragon.obj";

    SurfaceMesh input;
    printf("Loading %s...\n", path);
    read(input, path);
    auto bb = bounds(input);
    Scalar bb_diag = norm(bb.max() - bb.min());

    double mean_edge = 0;
    for (auto e : input.edges())
    {
        auto v0 = input.vertex(e, 0), v1 = input.vertex(e, 1);
        mean_edge += (double)distance(input.position(v0), input.position(v1));
    }
    mean_edge /= input.n_edges();
    printf("  %zu verts, %zu faces\n", input.n_vertices(), input.n_faces());
    printf("  bbox_diag=%.6f, mean_edge=%.6f\n\n", (double)bb_diag, mean_edge);

    RemeshingCudaParams p_cuda{};
    p_cuda.smooth_strategy = RemeshStrategy::CUDA;
    p_cuda.smooth_sub_iterations = 5;

    // Test targets that produce output meshes of varying sizes
    // Dragon mean_edge ≈ 0.000306, so:
    //   1.0x mean → ~438K verts (same density, just regularize)
    //   2.0x mean → ~110K verts (moderate coarsening)
    //   4.0x mean → ~28K verts  (heavy coarsening)
    //   8.0x mean → ~7K verts   (extreme coarsening)

    struct Case { const char* label; double factor; };
    Case cases[] = {
        {"1.0x mean (regularize)", 1.0},
        {"2.0x mean (moderate)",   2.0},
        {"4.0x mean (coarsen)",    4.0},
    };

    printf("============================================================\n");
    printf("  Stanford Dragon — CPU vs CUDA smooth (2 runs, best of)\n");
    printf("============================================================\n\n");
    printf("  %-24s %8s  %7s %7s  %7s %7s %7s\n",
           "Target", "edge", "verts", "faces", "CPU", "CUDA", "speedup");
    printf("  %-24s %8s  %7s %7s  %7s %7s %7s\n",
           "------------------------", "--------", "-------", "-------",
           "-------", "-------", "-------");

    for (auto& tc : cases)
    {
        Scalar target = (Scalar)(mean_edge * tc.factor);
        printf("  Running %-24s (target=%.6f)...\n", tc.label, (double)target);
        fflush(stdout);

        auto cpu = run(input, target, nullptr);
        auto cuda = run(input, target, &p_cuda);

        double cpu_best = std::min(cpu.times[0], cpu.times[1]);
        double cuda_best = std::min(cuda.times[0], cuda.times[1]);

        printf("  %-24s %8.6f  %7zu %7zu  %6.1fs %6.1fs %6.2fx\n",
               tc.label, (double)target, cuda.verts, cuda.faces,
               cpu_best, cuda_best, cpu_best / cuda_best);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
