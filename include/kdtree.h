#ifndef ICP_KDTREE_H_
#define ICP_KDTREE_H_

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

#include "pointcloud.h"

typedef struct {
	int point;   /* index into the source PointCloud */
	int axis;    /* split axis: 0=x, 1=y, 2=z */
	int left;    /* child node indices, or -1 */
	int right;
} KDNode;

typedef struct {
	const PointCloud *pts;   /* not owned */
	int    *perm;            /* working index permutation (owned) */
	KDNode *nodes;           /* node pool (owned) */
	int     n;
	int     root;            /* root node index, or -1 if empty */
} KDTree;

void kd_build(KDTree *t, const PointCloud *pts);
void kd_free(KDTree *t);

/* Nearest point to (qx,qy,qz); writes the point index and squared distance. */
void kd_nearest(const KDTree *t, float qx, float qy, float qz,
                int *best_idx, float *best_d2);

/* Brute-force equivalent, for benchmarking and as a correctness oracle. */
void bf_nearest(const PointCloud *pts, float qx, float qy, float qz,
                int *best_idx, float *best_d2);

#endif /* ICP_KDTREE_H_ */
