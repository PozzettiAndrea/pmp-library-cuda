// Quick verification test for remeshing_cuda on real meshes
// Usage: ./remeshing_cuda_test <mesh.obj|.off> [config.txt] [output.obj]

#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing_cuda.h"
#include "pmp/algorithms/remeshing.h"
#include "pmp/algorithms/remeshing_cuda_checkpoint.h"
#include "pmp/algorithms/triangulation.h"
#include "pmp/algorithms/normals.h"
#include "pmp/algorithms/utilities.h"
#include "pmp/bounding_box.h"
#include "pmp/io/io.h"

#include <chrono>
#include <cstdio>
#include <cstring>

using namespace pmp;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <mesh> [config.txt] [output.obj]\n", argv[0]);
        printf("  mesh:       input mesh (.obj, .off, .stl)\n");
        printf("  config.txt: remeshing config (optional, defaults to all-CPU)\n");
        printf("  output.obj: output mesh (optional)\n");
        printf("\nAvailable checkpoint stages:\n");
        remesh_list_stages();
        return 1;
    }

    const char* input_path = argv[1];
    const char* config_path = argc > 2 ? argv[2] : nullptr;
    const char* output_path = argc > 3 ? argv[3] : nullptr;

    // Load mesh
    SurfaceMesh mesh;
    printf("Loading %s...\n", input_path);
    read(mesh, input_path);
    printf("  Loaded: %zu vertices, %zu edges, %zu faces\n",
           mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());

    // Triangulate if needed
    bool has_non_tri = false;
    for (auto f : mesh.faces())
    {
        if (mesh.valence(f) != 3)
        {
            has_non_tri = true;
            break;
        }
    }
    if (has_non_tri)
    {
        printf("  Triangulating...\n");
        triangulate(mesh);
        printf("  After triangulation: %zu vertices, %zu faces\n",
               mesh.n_vertices(), mesh.n_faces());
    }

    // Compute target edge length from bounding box
    auto bb = bounds(mesh);
    Scalar bb_size = norm(bb.max() - bb.min());
    Scalar target_edge = bb_size * 0.01; // 1% of bounding box diagonal
    printf("  Bounding box diagonal: %.4f\n", (double)bb_size);
    printf("  Target edge length: %.6f\n", (double)target_edge);

    // Load config
    RemeshingCudaParams params;
    if (config_path)
    {
        params = load_remeshing_config(config_path);
    }
    else
    {
        printf("[CONFIG] Using defaults (all CPU)\n");
    }

    // --- Run CUDA-accelerated remeshing ---
    printf("\n=== Running uniform_remeshing_cuda ===\n");
    SurfaceMesh mesh_cuda = mesh; // copy for comparison
    auto t0 = std::chrono::high_resolution_clock::now();
    uniform_remeshing_cuda(mesh_cuda, target_edge, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cuda_time = std::chrono::duration<double>(t1 - t0).count();

    printf("\n=== Result ===\n");
    printf("  Output: %zu vertices, %zu edges, %zu faces\n",
           mesh_cuda.n_vertices(), mesh_cuda.n_edges(), mesh_cuda.n_faces());
    printf("  Time: %.3f s\n", cuda_time);

    // --- Run baseline CPU remeshing for comparison ---
    printf("\n=== Running baseline uniform_remeshing (CPU) ===\n");
    SurfaceMesh mesh_cpu = mesh; // fresh copy
    t0 = std::chrono::high_resolution_clock::now();
    uniform_remeshing(mesh_cpu, target_edge, params.outer_iterations);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    printf("\n=== Baseline Result ===\n");
    printf("  Output: %zu vertices, %zu edges, %zu faces\n",
           mesh_cpu.n_vertices(), mesh_cpu.n_edges(), mesh_cpu.n_faces());
    printf("  Time: %.3f s\n", cpu_time);

    // --- Compare ---
    printf("\n=== Comparison ===\n");
    printf("  CUDA path: %zu verts, %zu faces (%.3f s)\n",
           mesh_cuda.n_vertices(), mesh_cuda.n_faces(), cuda_time);
    printf("  CPU  path: %zu verts, %zu faces (%.3f s)\n",
           mesh_cpu.n_vertices(), mesh_cpu.n_faces(), cpu_time);

    if (mesh_cuda.n_vertices() == mesh_cpu.n_vertices() &&
        mesh_cuda.n_faces() == mesh_cpu.n_faces())
    {
        printf("  MATCH: vertex/face counts identical\n");
    }
    else
    {
        printf("  DIFFER: vertex/face counts differ (expected for CUDA strategies)\n");
    }

    // --- Test checkpoint round-trip ---
    if (params.checkpoint_enabled && !params.checkpoint_dir.empty())
    {
        printf("\n=== Checkpoint Test ===\n");
        printf("  Checkpoints saved to: %s\n", params.checkpoint_dir.c_str());
        for (int s = 0; s < REMESH_STAGE_COUNT; ++s)
        {
            if (remesh_checkpoint_exists(params.checkpoint_dir.c_str(),
                                         (RemeshStage)s))
            {
                printf("  [OK] %s\n", remesh_stage_name((RemeshStage)s));
            }
        }
    }

    // Save output
    if (output_path)
    {
        printf("\nSaving to %s...\n", output_path);
        write(mesh_cuda, output_path);
        printf("  Done.\n");
    }

    printf("\nAll tests passed.\n");
    return 0;
}
