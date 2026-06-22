#ifndef ICP_POINTCLOUD_H_
#define ICP_POINTCLOUD_H_

/*
 * Point cloud as a Struct-of-Arrays (separate x/y/z float arrays). The SoA
 * layout is deliberate: it is what lets the per-point transform and distance
 * kernels vectorize (SIMD) and coalesce nicely on the GPU later. Storage is
 * float; the geometric math (linalg.h) is done in double.
 */

#include "rng.h"

typedef struct {
	float *x, *y, *z;
	int    n;     /* number of valid points */
	int    cap;   /* allocated capacity */
} PointCloud;

void pc_init(PointCloud *c);
void pc_free(PointCloud *c);
void pc_reserve(PointCloud *c, int n);
void pc_push(PointCloud *c, float x, float y, float z);
void pc_copy(const PointCloud *src, PointCloud *dst);   /* deep copy */

/* Deep copy of `src` into `dst` with points reordered along a Morton (Z-order)
 * curve, so consecutive points are spatial neighbours (SIMD-batch locality). */
void pc_morton_order(const PointCloud *src, PointCloud *dst);

/* Axis-aligned bounding box of the cloud. */
void pc_bounds(const PointCloud *c, double bmin[3], double bmax[3]);

/*
 * Build a synthetic "scene" target cloud: points sampled on a room corner
 * (floor, ceiling, two perpendicular walls) plus two spheres. The mix of
 * surface normals gives full 6-DOF observability so ICP is well constrained.
 */
void pc_generate_scene(PointCloud *c, int n, rng_t *rng);

/*
 * Derive a source cloud from a target by applying a KNOWN rigid transform
 * (R, T), perturbing with Gaussian noise (noise_std), keeping only a random
 * fraction of points (keep_frac, models partial overlap), and adding a
 * fraction of uniform outliers (outlier_frac, so correspondence rejection
 * actually matters).
 */
void pc_make_source(const PointCloud *tgt, PointCloud *src,
                    const double R[9], const double T[3],
                    double noise_std, double keep_frac, double outlier_frac,
                    rng_t *rng);

/* In-place rigid transform: p <- R*p + T for every point. (SIMD kernel.) */
void pc_apply_transform(PointCloud *c, const double R[9], const double T[3]);

/* Write an ASCII .pcd file (same format the sibling project / tools expect). */
void pc_save_pcd(const PointCloud *c, const char *path, int stride);

#endif /* ICP_POINTCLOUD_H_ */
