// SPDX-License-Identifier: MIT

// ============================================================
// CUDA kernels for edge collapsing (Phase 5 — deferred)
//
// This is the most complex operation to parallelize due to:
//   - Link-condition topology checks
//   - Large conflict zones (two-ring neighborhood)
//   - Must check collapse doesn't create too-long edges
//   - Requires GPU compaction (replaces garbage_collection)
//
// Currently uses CPU fallback. CUDA implementation is a stretch goal.
// ============================================================

#include "gpu_trimesh.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace pmp {

// TODO: Phase 5 implementation (stretch goal)

} // namespace pmp
