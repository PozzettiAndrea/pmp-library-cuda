// SPDX-License-Identifier: MIT

#include "pmp/algorithms/remeshing_cuda_checkpoint.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>

namespace pmp {

// ============================================================
// Stage name mapping
// ============================================================

static const char* stage_names[] = {
    "preprocessed",
    "post-split",
    "post-collapse",
    "post-flip",
    "post-smooth",
    "complete",
};

RemeshStage remesh_stage_from_name(const char* name)
{
    for (int i = 0; i < REMESH_STAGE_COUNT; ++i)
    {
        if (strcmp(name, stage_names[i]) == 0)
            return (RemeshStage)i;
    }
    return REMESH_STAGE_NONE;
}

const char* remesh_stage_name(RemeshStage s)
{
    if (s >= 0 && s < REMESH_STAGE_COUNT)
        return stage_names[s];
    return "unknown";
}

void remesh_list_stages()
{
    printf("Remeshing pipeline stages:\n");
    for (int i = 0; i < REMESH_STAGE_COUNT; ++i)
    {
        printf("  %d: %s\n", i, stage_names[i]);
    }
}

// ============================================================
// Checkpoint file path
// ============================================================

static std::string checkpoint_path(const char* dir, RemeshStage stage)
{
    return std::string(dir) + "/" + stage_names[stage] + ".rmc";
}

bool remesh_checkpoint_exists(const char* dir, RemeshStage stage)
{
    struct stat st;
    std::string path = checkpoint_path(dir, stage);
    return stat(path.c_str(), &st) == 0;
}

// ============================================================
// Serialization helpers (inline, matching qw_serialize.h)
// ============================================================

namespace ser {

static void Save(FILE* fp, int32_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void Read(FILE* fp, int32_t& v) { size_t r = fread(&v, sizeof(v), 1, fp); (void)r; }

static void Save(FILE* fp, bool v) { int32_t i = v ? 1 : 0; Save(fp, i); }
static void Read(FILE* fp, bool& v) { int32_t i; Read(fp, i); v = (i != 0); }

// POD vectors
static void Save(FILE* fp, const std::vector<int32_t>& v)
{
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    if (n > 0) fwrite(v.data(), sizeof(int32_t), n, fp);
}
static void Read(FILE* fp, std::vector<int32_t>& v)
{
    int32_t n;
    Read(fp, n);
    v.resize(n);
    if (n > 0) { size_t r = fread(v.data(), sizeof(int32_t), n, fp); (void)r; }
}

static void Save(FILE* fp, const std::vector<float>& v)
{
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    if (n > 0) fwrite(v.data(), sizeof(float), n, fp);
}
static void Read(FILE* fp, std::vector<float>& v)
{
    int32_t n;
    Read(fp, n);
    v.resize(n);
    if (n > 0) { size_t r = fread(v.data(), sizeof(float), n, fp); (void)r; }
}

} // namespace ser

// ============================================================
// Save checkpoint
// ============================================================

void save_remesh_checkpoint(const RemeshCheckpointData& data,
                            RemeshStage stage, const char* dir,
                            const RemeshCheckpointHeader& hdr_in)
{
    mkdir(dir, 0755);

    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        printf("[CHECKPOINT] ERROR: Cannot open %s for writing\n",
               path.c_str());
        return;
    }

    // Write header
    RemeshCheckpointHeader hdr = hdr_in;
    memcpy(hdr.magic, "RMC", 4);
    hdr.version = 1;
    strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);
    hdr.timestamp = (int64_t)time(nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);

    // Write stage index
    int32_t stage_idx = (int32_t)stage;
    ser::Save(fp, stage_idx);

    // ---- Always save: mesh geometry ----
    ser::Save(fp, data.positions);
    ser::Save(fp, data.faces);

    // ---- Per-vertex properties ----
    ser::Save(fp, data.vsizing);
    ser::Save(fp, data.vlocked);
    ser::Save(fp, data.vfeature);

    // ---- Per-edge properties ----
    ser::Save(fp, data.efeature);
    ser::Save(fp, data.elocked);

    // ---- Reference mesh (for projection) ----
    ser::Save(fp, data.has_reference);
    if (data.has_reference)
    {
        ser::Save(fp, data.ref_positions);
        ser::Save(fp, data.ref_faces);
        ser::Save(fp, data.ref_normals);
        ser::Save(fp, data.ref_sizing);
    }

    fclose(fp);

    // Print summary
    long file_size = 0;
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        file_size = st.st_size;
    printf("[CHECKPOINT] Saved '%s' to %s (%.1f MB)\n", stage_names[stage],
           path.c_str(), file_size / (1024.0 * 1024.0));
    printf("[CHECKPOINT]   mesh: %d verts, %d faces | iter=%d\n",
           hdr.num_vertices, hdr.num_faces, hdr.iteration);
    printf("[CHECKPOINT]   strategies: split=%d collapse=%d flip=%d smooth=%d\n",
           hdr.split_strategy, hdr.collapse_strategy, hdr.flip_strategy,
           hdr.smooth_strategy);
}

// ============================================================
// Load checkpoint
// ============================================================

RemeshStage load_remesh_checkpoint(RemeshCheckpointData& data, const char* dir,
                                   RemeshStage stage)
{
    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        printf("[CHECKPOINT] ERROR: Cannot open %s for reading\n",
               path.c_str());
        return REMESH_STAGE_NONE;
    }

    // Read and validate header
    RemeshCheckpointHeader hdr;
    size_t r = fread(&hdr, sizeof(hdr), 1, fp);
    (void)r;
    if (memcmp(hdr.magic, "RMC", 3) != 0)
    {
        printf("[CHECKPOINT] ERROR: Invalid magic in %s (expected RMC)\n",
               path.c_str());
        fclose(fp);
        return REMESH_STAGE_NONE;
    }
    if (hdr.version != 1)
    {
        printf("[CHECKPOINT] ERROR: Unsupported version %d in %s\n",
               hdr.version, path.c_str());
        fclose(fp);
        return REMESH_STAGE_NONE;
    }

    int32_t stage_idx;
    ser::Read(fp, stage_idx);
    RemeshStage saved_stage = (RemeshStage)stage_idx;

    printf("[CHECKPOINT] Loading '%s' from %s\n", stage_names[saved_stage],
           path.c_str());
    printf("[CHECKPOINT]   mesh: %d verts, %d faces | iter=%d\n",
           hdr.num_vertices, hdr.num_faces, hdr.iteration);

    // ---- Mesh geometry ----
    ser::Read(fp, data.positions);
    ser::Read(fp, data.faces);

    // ---- Per-vertex properties ----
    ser::Read(fp, data.vsizing);
    ser::Read(fp, data.vlocked);
    ser::Read(fp, data.vfeature);

    // ---- Per-edge properties ----
    ser::Read(fp, data.efeature);
    ser::Read(fp, data.elocked);

    // ---- Reference mesh ----
    ser::Read(fp, data.has_reference);
    if (data.has_reference)
    {
        ser::Read(fp, data.ref_positions);
        ser::Read(fp, data.ref_faces);
        ser::Read(fp, data.ref_normals);
        ser::Read(fp, data.ref_sizing);
    }

    fclose(fp);
    return saved_stage;
}

} // namespace pmp
