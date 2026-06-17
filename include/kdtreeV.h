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

typedef float v8f __attribute__((vector_size(8*sizeof(float))));
typedef float v8f_u __attribute__((vector_size(8*sizeof(float)), aligned(4)));
typedef int v8i __attribute__((vector_size(32)));	// same width of v8f

// NOTE: this implementation is not memory efficient, 
// implement a union for better memory management
typedef struct {
	/* leaf */
	float xs[8], ys[8], zs[8];	// arrays of points' coordinates
	int idx[8];	/* lane -> target point index */
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

#endif /* ICP_KDTREEV_H_ */
