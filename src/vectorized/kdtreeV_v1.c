/*
 * kdtreeV.c -- median-split 3D KD-tree with KD_W-wide leaf buckets, plus a
 * brute-force fallback. Vectorized with GCC/clang vector extensions; the width
 * KD_W (8 on AVX, 4 on NEON) is chosen in kdtreeV.h from the target arch.
 */

#include "kdtreeV.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

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
	if ((hi - lo) <= KD_W)	/* leaf: 1..KD_W points */
	{
		int node = (*next)++;
		int count = hi - lo;
		KDNodeV *nd = &t->nodes[node];

		for (int i = 0; i < count; i++)
		{
			int p = t->perm[lo + i];
			nd->xs[i]  = t->pts->x[p];
			nd->ys[i]  = t->pts->y[p];
			nd->zs[i]  = t->pts->z[p];
			nd->idx[i] = p;
		}
		/* Fill dead lanes with a real point (lane 0): their distances are
		 * harmless duplicates that can never spuriously beat the nearest. */
		for (int i = count; i < KD_W; i++)
		{
			nd->xs[i]  = nd->xs[0];
			nd->ys[i]  = nd->ys[0];
			nd->zs[i]  = nd->zs[0];
			nd->idx[i] = nd->idx[0];
		}
		nd->count = count;
		return node;
	}

	/* internal node: pure divider, holds no points */
	int axis = depth % 3;
	int mid  = lo + (hi - lo) / 2;
	qselect(t, lo, hi - 1, mid, axis);

	int node = (*next)++;
	t->nodes[node].count = -1;
	t->nodes[node].axis  = axis;
	t->nodes[node].split = coord(t->pts, t->perm[mid], axis);  /* read before recursing */
	t->nodes[node].left  = build_rec(t, next, lo, mid, depth + 1);
	t->nodes[node].right = build_rec(t, next, mid, hi, depth + 1);  /* mid stays on the right */
	return node;
}

static void flatten_bfs(KDTreeV *t, int n_nodes)
{
	int *queue = malloc((size_t)n_nodes * sizeof *queue);
	int *newidx = malloc((size_t)n_nodes * sizeof *newidx);
	if (!queue || !newidx) { perror("flatten_bfs"); exit(1); }

	int head = 0, tail = 0;
	queue[tail++] = t->root;
	while (head < tail)
	{
		int old = queue[head];
		newidx[old] = head;
		head++;
		if (t->nodes[old].count < 0)
		{
			queue[tail++] = t->nodes[old].left;
			queue[tail++] = t->nodes[old].right;
		}
	}

	KDNodeV *neu = malloc((size_t)n_nodes * sizeof *neu);
	if (!neu) { perror("flatten_bfs"); exit(1); }
	for (int i = 0; i < n_nodes; i++)
	{
		KDNodeV nd = t->nodes[queue[i]];
		if (nd.count < 0)
		{
			nd.left = newidx[nd.left];
			nd.right = newidx[nd.right];
		}
		neu[i] = nd;
	}

	free(t->nodes);
	t->nodes = neu;
	t->root = 0;
	free(queue);
	free(newidx);
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
	t->nodes = malloc((size_t)pts->n * sizeof *t->nodes);  /* over-allocation: total nodes < n */
	if (!t->perm || !t->nodes) { perror("kd_build"); exit(1); }
	for (int i = 0; i < pts->n; i++) t->perm[i] = i;

	int next = 0;
	t->root = build_rec(t, &next, 0, pts->n, 0);
	flatten_bfs(t, next);
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

static void nn_search(const KDTreeV *t, float qx, float qy, float qz,
                   int *best_idx, float *best_d2)
{
	int stack[64];	
	float bound[64];
	int sp = 0;
	stack[sp] = t->root; bound[sp] = 0.0f; sp++;

	while (sp > 0) 
	{
		sp--;
		if (bound[sp] >= *best_d2) continue;
		const KDNodeV *nd = &t->nodes[stack[sp]];

		/* leaf: one SIMD distance op over the bucket, then a local reduction */
		if (nd->count >= 0)
		{
			vf xs = *(const vf_u *)nd->xs;
			vf ys = *(const vf_u *)nd->ys;
			vf zs = *(const vf_u *)nd->zs;
			vf dx = xs - qx;	/* qx broadcast across lanes */
			vf dy = ys - qy;
			vf dz = zs - qz;
			vf d2 = (dx * dx) + (dy * dy) + (dz * dz);

			float dd[KD_W];
			memcpy(dd, &d2, sizeof dd);
			int lane = 0;
			for (int i = 1; i < KD_W; i++)
				if (dd[i] < dd[lane]) lane = i;

			if (dd[lane] < *best_d2)	/* scalar compare against running radius */
			{
				*best_idx = nd->idx[lane];
				*best_d2  = dd[lane];
			}
			continue;
		}

		/* internal */
		int   ax   = nd->axis;
		float q    = (ax == 0) ? qx : (ax == 1) ? qy : qz;
		float diff = q - nd->split;

		int near = diff < 0 ? nd->left  : nd->right;
		int far  = diff < 0 ? nd->right : nd->left;
		
		stack[sp] = far; 
		bound[sp] = diff*diff; 
		sp++;
		stack[sp] = near;
		bound[sp] = 0.0f;
		sp++;
	}
}

void kd_nearest(const KDTreeV *t, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	*best_idx = -1;
	*best_d2  = FLT_MAX;
	nn_search(t, qx, qy, qz, best_idx, best_d2);
}

void bf_nearest(const PointCloud *pts, float qx, float qy, float qz,
                int *best_idx, float *best_d2)
{
	if (pts->n % KD_W != 0)
	{
		fprintf(stderr, "Error: point cloud size (%d) must be a multiple of %d\n",
		        pts->n, KD_W);
		exit(EXIT_FAILURE);
	}

	vf bd;
	vi bi;
	for (int k = 0; k < KD_W; k++) { bd[k] = FLT_MAX; bi[k] = -1; }

	for (int i = 0; i < pts->n; i += KD_W) {
		vf xs = *(const vf_u *)(pts->x + i);
		vf ys = *(const vf_u *)(pts->y + i);
		vf zs = *(const vf_u *)(pts->z + i);
		vf dx = xs - qx;	/* broadcast */
		vf dy = ys - qy;
		vf dz = zs - qz;
		vf d2 = (dx * dx) + (dy * dy) + (dz * dz);

		vi mask = (d2 < bd);
		vi idx;
		for (int k = 0; k < KD_W; k++) idx[k] = i + k;

		bd = (vf)((mask & (vi)d2) | (~mask & (vi)bd));	/* (vi)d2 is a bit reinterpretation */
		bi = (mask & idx) | (~mask & bi);
	}

	float dd[KD_W];
	memcpy(dd, &bd, sizeof dd);
	int lane = 0;
	for (int k = 1; k < KD_W; k++)
		if (dd[k] < dd[lane]) lane = k;

	*best_idx = bi[lane];
	*best_d2  = dd[lane];
}
