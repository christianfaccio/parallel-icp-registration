#ifndef ICP_KDTREEV_H_
#define ICP_KDTREEV_H_

#include <string.h>
#include "pointcloud.h"

#define KD_W 16

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
	int     n_nodes;         /* number of valid entries in `nodes` */
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
