// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// SPDX-License-Identifier: MIT

#pragma once

#include "pmp/surface_mesh.h"

#include <memory>

namespace pmp {

//! \brief Perform uniform remeshing.
//! \details Performs incremental remeshing based
//! on edge collapse, split, flip, and tangential relaxation.
//! See \cite botsch_2004_remeshing and \cite dunyach_2013_adaptive for details.
//! \param mesh The input mesh, modified in place.
//! \param edge_length The target edge length.
//! \param iterations The number of iterations
//! \param use_projection Use back-projection to the input surface.
//! \pre Input mesh needs to be a triangle mesh.
//! \throw InvalidInputException if the input precondition is violated.
//! \ingroup algorithms
void uniform_remeshing(SurfaceMesh& mesh, Scalar edge_length,
                       unsigned int iterations = 10,
                       bool use_projection = true);

//! \brief Perform adaptive remeshing.
//! \details Performs incremental remeshing based
//! on edge collapse, split, flip, and tangential relaxation.
//! See \cite botsch_2004_remeshing and \cite dunyach_2013_adaptive for details.
//! \param mesh The input mesh, modified in place.
//! \param min_edge_length The minimum edge length.
//! \param max_edge_length The maximum edge length.
//! \param approx_error The maximum approximation error.
//! \param iterations The number of iterations.
//! \param use_projection Use back-projection to the input surface.
//! \pre Input mesh needs to be a triangle mesh.
//! \throw InvalidInputException if the input precondition is violated.
//! \ingroup algorithms
void adaptive_remeshing(SurfaceMesh& mesh, Scalar min_edge_length,
                        Scalar max_edge_length, Scalar approx_error,
                        unsigned int iterations = 10,
                        bool use_projection = true);

// Forward declaration (implementation detail)
class TriangleKdTree;

//! \brief Remeshing engine exposing individual operations.
//! \details Allows calling split, collapse, flip, smooth individually
//! for per-operation strategy dispatch (CPU vs CUDA).
class Remeshing
{
public:
    Remeshing(SurfaceMesh& mesh);
    ~Remeshing();

    // Full pipeline (original API, calls all operations)
    void uniform_remeshing(Scalar edge_length, unsigned int iterations = 10,
                           bool use_projection = true);
    void adaptive_remeshing(Scalar min_edge_length, Scalar max_edge_length,
                            Scalar approx_error, unsigned int iterations = 10,
                            bool use_projection = true);

    // Individual operations (for per-operation dispatch)
    void preprocessing();
    void postprocessing();
    void split_long_edges();
    void collapse_short_edges();
    void flip_edges();
    void tangential_smoothing(unsigned int iterations);
    void remove_caps();

    // Setup (must be called before individual operations)
    void set_uniform(Scalar edge_length, bool use_projection);
    void set_adaptive(Scalar min_edge_length, Scalar max_edge_length,
                      Scalar approx_error, bool use_projection);

private:
    Point minimize_squared_areas(Vertex v);
    Point weighted_centroid(Vertex v);
    void project_to_reference(Vertex v);

    bool is_too_long(Vertex v0, Vertex v1) const
    {
        return distance(points_[v0], points_[v1]) >
               4.0 / 3.0 * std::min(vsizing_[v0], vsizing_[v1]);
    }
    bool is_too_short(Vertex v0, Vertex v1) const
    {
        return distance(points_[v0], points_[v1]) <
               4.0 / 5.0 * std::min(vsizing_[v0], vsizing_[v1]);
    }

    SurfaceMesh& mesh_;
    std::shared_ptr<SurfaceMesh> refmesh_;

    bool use_projection_;
    std::unique_ptr<TriangleKdTree> kd_tree_;

    bool uniform_;
    Scalar target_edge_length_;
    Scalar min_edge_length_;
    Scalar max_edge_length_;
    Scalar approx_error_;

    bool has_feature_vertices_{false};
    bool has_feature_edges_{false};
    VertexProperty<Point> points_;
    VertexProperty<Point> vnormal_;
    VertexProperty<bool> vfeature_;
    EdgeProperty<bool> efeature_;
    VertexProperty<bool> vlocked_;
    EdgeProperty<bool> elocked_;
    VertexProperty<Scalar> vsizing_;

    VertexProperty<Point> refpoints_;
    VertexProperty<Point> refnormals_;
    VertexProperty<Scalar> refsizing_;
};

} // namespace pmp
