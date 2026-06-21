#ifndef ICP_KDTREEV_H_
#define ICP_KDTREEV_H_

/*
 * Nearest-neighbour search over a target cloud -- the dominant ICP cost and the
 * headline parallel bottleneck.
 *
 * Two backends are provided:
 *   - bf_nearest : brute-force O(M) linear scan. Trivially parallel and used as
 *                  the correctness oracle for the tree.
 *   - kd_*       : a median-split 3D KD-tree, O(log M) average per query. The
 *                  recursive, index-chasing traversal is exactly the structure
 *                  whose poor cache locality / branch divergence the later
 *                  parallel phases attack (flattened tree, voxel grid, ...).
 */

#include <string.h>
#include "pointcloud.h"

/*
 * SIMD width KD_W = lanes per node bucket / per brute-force step.
 *   x86 AVX  -> 256-bit registers -> 8 floats
 *   ARM NEON -> 128-bit registers -> 4 floats
 * The whole algorithm is width-generic; this is the ONLY place the x86 and
 * ARM "versions" differ. Selected automatically from the target arch.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
	#define KD_W 4
#else
	#define KD_W 16
#endif

typedef float vf   __attribute__((vector_size(KD_W * sizeof(float))));
typedef float vf_u __attribute__((vector_size(KD_W * sizeof(float)), aligned(4)));
typedef int   vi   __attribute__((vector_size(KD_W * sizeof(int))));  /* same width as vf */

// NOTE: this implementation is not memory efficient,
// implement a union for better memory management
typedef struct {
	/* leaf */
	float xs[KD_W], ys[KD_W], zs[KD_W];	// arrays of points' coordinates
	int idx[KD_W];	/* lane -> target point index */
	int count;	/* -1: internal */

	/* internal */
	int axis;    /* split axis: 0=x, 1=y, 2=z */
	float split; /* routing threshold */
	int left;    /* child node indices, or -1 */
	int right;
} KDNodeV;

typedef struct {
	const PointCloud *pts;   /* not owned */
	int    *perm;            /* working index permutation (owned) */
	KDNodeV *nodes;           /* node pool (owned) */
	int     n;
	int     root;            /* root node index, or -1 if empty */
} KDTreeV;

void kd_build(KDTreeV *t, const PointCloud *pts);
void kd_free(KDTreeV *t);

/* Nearest point to (qx,qy,qz); writes the point index and squared distance. */
void kd_nearest(const KDTreeV *t, float qx, float qy, float qz,
                int *best_idx, float *best_d2);

/* Brute-force equivalent, for benchmarking and as a correctness oracle. */
void bf_nearest(const PointCloud *pts, float qx, float qy, float qz,
                int *best_idx, float *best_d2);

/* Nearest point to (qx,qy,qz); writes the point index and squared distance. */
void kd_nearest_simd(const KDTreeV *t, vf qx, vf qy, vf qz,
                vi *best_idx, vf *best_d2);

/* Brute-force equivalent, for benchmarking and as a correctness oracle. */
void bf_nearest_simd(const PointCloud *pts, vf qx, vf qy, vf qz,
                vi *best_idx, vf *best_d2);


#endif /* ICP_KDTREEV_H_ */
