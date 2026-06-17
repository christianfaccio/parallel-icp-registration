/*
 * kdtree.c -- median-split 3D KD-tree plus a brute-force fallback.
 */

#include "kdtreeV.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined (__AVX__) || defined (__AVX2__)
#include <immintrin.h>
#endif

static inline float coord(const PointCloud *p, int i, int axis)
{
	return axis == 0 ? p->x[i] : axis == 1 ? p->y[i] : p->z[i];
}

static inline void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

/* Lomuto partition of perm[lo..hi] (inclusive) on `axis`; pivot = perm[hi]. */
static int partition(KDTreeV *t, int lo, int hi, int axis)
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
static void qselect(KDTreeV *t, int lo, int hi, int k, int axis)
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
static int build_rec(KDTreeV *t, int *next, int lo, int hi, int depth)
{
	if ((hi - lo) <= 8)	// single leaf (1-8 points)
	{
		int node = (*next)++;
		int count = hi - lo;
		
		for (int i = 0; i < count; i++)
		{
			int p = t->perm[lo+i];
			t->nodes[node].xs[i] = t->pts->x[p];
			t->nodes[node].ys[i] = t->pts->y[p];
			t->nodes[node].zs[i] = t->pts->z[p];
			t->nodes[node].idx[i] = p;
		}
		t->nodes[node].count = count;
		return node;
	} else
	{
		int axis = depth % 3;
		int mid = lo + (hi - lo) / 2;
		qselect(t, lo, hi - 1, mid, axis);
		int pmid = t->perm[mid];
		float split = (axis == 0) ? t->pts->x[pmid]
			    : (axis == 1) ? t->pts->x[pmid]
			    : t->pts->x[pmid];
		int node = (*next)++;


		t->nodes[node].count = -1;
		t->nodes[node].split = split;	
		t->nodes[node].axis  = axis;
		t->nodes[node].left  = build_rec(t, next, lo, mid, depth + 1);
		t->nodes[node].right = build_rec(t, next, mid, hi, depth + 1);
		return node;
	}
}

void kd_build(KDTreeV *t, const PointCloud *pts)
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

void kd_free(KDTreeV *t)
{
	free(t->perm);
	free(t->nodes);
	t->perm = NULL;
	t->nodes = NULL;
	t->n = 0;
	t->root = -1;
}

static void nn_rec(const KDTreeV *t, int node, float qx, float qy, float qz,
                   v8i *best_idx, v8f *best_d2)
{
	if (node < 0) return;
	const KDNodeV *nd = &t->nodes[node];
	
	/* leaf */
	if (nd->count >= 0)
	{
		v8f xs = *(const v8f_u *)(&nd->xs);
		v8f dx = xs - qx;
		v8f ys = *(const v8f_u *)(&nd->ys);
		v8f dy = ys - qy;
		v8f zs = *(const v8f_u *)(&nd->zs);
		v8f dz = zs - qz;
		v8f d2 = (dx*dx) + (dy*dy) + (dz*dz);

		float dd[8];
		memcpy(dd, &bd, 32);
		int lane = 0;
		for (int i = 0; i < 8; i++)
		{
			if (dd[i] < dd[lane])
			{
				lane = i;
			}
		}
		if (dd[lane] < *best_d2)
		{
			*best_idx = nd->idx[lane];
			*best_d2 = dd[lane];
		}
		return;
	}

	/* internal */
	int   ax    = nd->axis;
	float q     = (ax == 0) ? qx : (ax == 1) ? qy : qz;
	float diff  = q - nd->split;

	int near = diff < 0 ? nd->left  : nd->right;
	int far  = diff < 0 ? nd->right : nd->left;

	nn_rec(t, near, qx, qy, qz, best_idx, best_d2);
	if (diff * diff < *best_d2)            /* the far side might still hold closer */
		nn_rec(t, far, qx, qy, qz, best_idx, best_d2);
}

void kd_nearest(const KDTreeV *t, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	*best_idx = -1;
	*best_d2  = FLT_MAX;
	nn_rec(t, t->root, qx, qy, qz, best_idx, best_d2);
}

void bf_nearest(const PointCloud *pts, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	if (pts->n % 8 != 0)
	{
		fprintf(stderr, "Error: point cloud size (%d) must be a multiple of 8\n", pts->n);
		exit(EXIT_FAILURE);
	}
	v8i bi = {-1, -1, -1, -1, -1, -1, -1, -1};
	v8f bd = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}; 
	
	for (int i = 0; i < pts->n; i+=8) {
		v8f xs = *(const v8f_u *)(pts->x + i);
		v8f dx = xs - qx;	// broadcast
		v8f ys = *(const v8f_u *)(pts->y + i);
		v8f dy = ys - qy;	// broadcast
		v8f zs = *(const v8f_u *)(pts->z + i);
		v8f dz = zs - qz;	// broadcast
		v8f d2 = (dx*dx) + (dy*dy) + (dz*dz);

		v8i mask = (d2 < bd);
		v8i idx = {i+0, i+1, i+2, i+3, i+4, i+5, i+6, i+7};
		bd = (v8f)((mask & (v8i)d2) | (~mask & (v8i)bd));	// (v8i)d is a bit reinterpretation
		bi = (mask & idx) | (~mask & bi);
	}
	float dd[8];
	memcpy(dd, &bd, 32);
	int lane = 0;
	for (int k = 1; k < 8; k++)
	{
		if (dd[k] < dd[lane])
		{
			lane = k;
		}
	}
	*best_idx = bi[lane];
	*best_d2  = dd[lane];
}
