/*
 * kdtree.c -- median-split 3D KD-tree plus a brute-force fallback.
 */

#include "kdtree.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

static inline float coord(const PointCloud *p, int i, int axis)
{
	return axis == 0 ? p->x[i] : axis == 1 ? p->y[i] : p->z[i];
}

static inline void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

/* Lomuto partition of perm[lo..hi] (inclusive) on `axis`; pivot = perm[hi]. */
static int partition(KDTree *t, int lo, int hi, int axis)
{
	float pv = coord(t->pts, t->perm[hi], axis);
	int i = lo;
	for (int j = lo; j < hi; j++)
		if (coord(t->pts, t->perm[j], axis) < pv)
			swap_int(&t->perm[i++], &t->perm[j]);
	swap_int(&t->perm[i], &t->perm[hi]);
	return i;
}

/* Quickselect: reorder perm[lo..hi] so perm[k] holds the k-th smallest on axis. */
static void qselect(KDTree *t, int lo, int hi, int k, int axis)
{
	while (lo < hi) {
		/* mid pivot guards against the sorted-input worst case */
		swap_int(&t->perm[lo + (hi - lo) / 2], &t->perm[hi]);
		int p = partition(t, lo, hi, axis);
		if (p == k) return;
		else if (p < k) lo = p + 1;
		else            hi = p - 1;
	}
}

/* Recursively build over perm[lo,hi); returns the node index or -1. */
static int build_rec(KDTree *t, int *next, int lo, int hi, int depth)
{
	if (lo >= hi) return -1;
	int axis = depth % 3;
	int mid = lo + (hi - lo) / 2;
	qselect(t, lo, hi - 1, mid, axis);

	int node = (*next)++;
	t->nodes[node].point = t->perm[mid];
	t->nodes[node].axis  = axis;
	t->nodes[node].left  = build_rec(t, next, lo, mid, depth + 1);
	t->nodes[node].right = build_rec(t, next, mid + 1, hi, depth + 1);
	return node;
}

void kd_build(KDTree *t, const PointCloud *pts)
{
	t->pts = pts;
	t->n   = pts->n;
	t->root = -1;
	t->perm  = NULL;
	t->nodes = NULL;
	if (pts->n == 0) return;

	t->perm  = malloc((size_t)pts->n * sizeof *t->perm);
	t->nodes = malloc((size_t)pts->n * sizeof *t->nodes);
	if (!t->perm || !t->nodes) { perror("kd_build"); exit(1); }
	for (int i = 0; i < pts->n; i++) t->perm[i] = i;

	int next = 0;
	t->root = build_rec(t, &next, 0, pts->n, 0);
}

void kd_free(KDTree *t)
{
	free(t->perm);
	free(t->nodes);
	t->perm = NULL;
	t->nodes = NULL;
	t->n = 0;
	t->root = -1;
}

static void nn_rec(const KDTree *t, int node, float qx, float qy, float qz,
                   int *best_idx, float *best_d2)
{
	if (node < 0) return;
	const KDNode *nd = &t->nodes[node];
	int p = nd->point;

	float dx = t->pts->x[p] - qx;
	float dy = t->pts->y[p] - qy;
	float dz = t->pts->z[p] - qz;
	float d2 = dx*dx + dy*dy + dz*dz;
	if (d2 < *best_d2) { *best_d2 = d2; *best_idx = p; }

	int   ax    = nd->axis;
	float q     = ax == 0 ? qx : ax == 1 ? qy : qz;
	float split = coord(t->pts, p, ax);
	float diff  = q - split;

	int near = diff < 0 ? nd->left  : nd->right;
	int far  = diff < 0 ? nd->right : nd->left;

	nn_rec(t, near, qx, qy, qz, best_idx, best_d2);
	if (diff * diff < *best_d2)            /* the far side might still hold closer */
		nn_rec(t, far, qx, qy, qz, best_idx, best_d2);
}

void kd_nearest(const KDTree *t, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	*best_idx = -1;
	*best_d2  = FLT_MAX;
	nn_rec(t, t->root, qx, qy, qz, best_idx, best_d2);
}

void bf_nearest(const PointCloud *pts, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	int   bi = -1;
	float bd = FLT_MAX;
	for (int i = 0; i < pts->n; i++) {
		float dx = pts->x[i] - qx;
		float dy = pts->y[i] - qy;
		float dz = pts->z[i] - qz;
		float d2 = dx*dx + dy*dy + dz*dz;
		if (d2 < bd) { bd = d2; bi = i; }
	}
	*best_idx = bi;
	*best_d2  = bd;
}
