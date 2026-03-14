// Per-iteration per-operation profiler for Dragon at 1x mean edge
// Shows exactly where time is spent in both CPU and CUDA paths

#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing_cuda.h"
#include "pmp/algorithms/remeshing.h"
#include "pmp/algorithms/normals.h"
#include "pmp/algorithms/utilities.h"
#include "pmp/bounding_box.h"
#include "pmp/io/io.h"

#include <chrono>
#include <cstdio>

using namespace pmp;
using Clock = std::chrono::high_resolution_clock;

static double elapsed(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1]
        : "../../../cudageom/QuadriFlow-cuda/examples/dragon.obj";

    SurfaceMesh input;
    printf("Loading %s...\n", path);
    read(input, path);

    double mean_edge = 0;
    for (auto e : input.edges())
    {
        auto v0 = input.vertex(e, 0), v1 = input.vertex(e, 1);
        mean_edge += (double)distance(input.position(v0), input.position(v1));
    }
    mean_edge /= input.n_edges();

    Scalar target = (Scalar)mean_edge; // 1x mean
    printf("  %zu verts, %zu faces, target_edge=%.6f (1x mean)\n\n",
           input.n_vertices(), input.n_faces(), (double)target);

    // ====== CPU PROFILE ======
    printf("====== CPU per-iteration profile ======\n\n");
    {
        SurfaceMesh mesh = input;
        Remeshing engine(mesh);
        engine.set_uniform(target, true);

        auto t0 = Clock::now();
        engine.preprocessing();
        auto t1 = Clock::now();
        printf("  preprocessing:  %6.2fs  (%zu v, %zu f)\n",
               elapsed(t0, t1), mesh.n_vertices(), mesh.n_faces());

        printf("\n  %4s  %8s %8s  %7s %7s %7s %7s %7s  %7s\n",
               "iter", "verts", "faces", "split", "norms", "collap", "flip", "smooth", "total");

        double total_split = 0, total_norms = 0, total_coll = 0, total_flip = 0, total_smooth = 0;

        for (int i = 0; i < 10; ++i)
        {
            auto is = Clock::now();
            engine.split_long_edges();
            auto ie1 = Clock::now();

            vertex_normals(mesh);
            auto ie2 = Clock::now();

            engine.collapse_short_edges();
            auto ie3 = Clock::now();

            engine.flip_edges();
            auto ie4 = Clock::now();

            engine.tangential_smoothing(5);
            auto ie5 = Clock::now();

            double dt_s = elapsed(is, ie1);
            double dt_n = elapsed(ie1, ie2);
            double dt_c = elapsed(ie2, ie3);
            double dt_f = elapsed(ie3, ie4);
            double dt_m = elapsed(ie4, ie5);
            double dt_t = elapsed(is, ie5);

            total_split += dt_s; total_norms += dt_n;
            total_coll += dt_c; total_flip += dt_f; total_smooth += dt_m;

            printf("  %4d  %8zu %8zu  %6.2fs %6.2fs %6.2fs %6.2fs %6.2fs  %6.2fs\n",
                   i, mesh.n_vertices(), mesh.n_faces(),
                   dt_s, dt_n, dt_c, dt_f, dt_m, dt_t);
        }

        auto t2 = Clock::now();
        engine.remove_caps();
        engine.postprocessing();
        auto t3 = Clock::now();

        double grand = elapsed(t0, t3);
        printf("\n  TOTALS:                      %6.2fs %6.2fs %6.2fs %6.2fs %6.2fs\n",
               total_split, total_norms, total_coll, total_flip, total_smooth);
        printf("  postprocessing: %.2fs\n", elapsed(t2, t3));
        printf("  GRAND TOTAL:    %.2fs\n", grand);
        printf("  Output: %zu verts, %zu faces\n\n", mesh.n_vertices(), mesh.n_faces());
    }

    // ====== CUDA PROFILE ======
    printf("====== CUDA smooth per-iteration profile ======\n\n");
    {
        RemeshingCudaParams p{};
        p.smooth_strategy = RemeshStrategy::CUDA;
        p.smooth_sub_iterations = 5;

        SurfaceMesh mesh = input;
        auto t0 = Clock::now();
        uniform_remeshing_cuda(mesh, target, p);
        auto t1 = Clock::now();
        printf("  GRAND TOTAL:    %.2fs\n", elapsed(t0, t1));
        printf("  Output: %zu verts, %zu faces\n", mesh.n_vertices(), mesh.n_faces());
    }

    return 0;
}
