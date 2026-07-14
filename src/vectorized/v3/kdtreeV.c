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

/* Portable horizontal reductions over a vf. GCC has no __builtin_reduce_*
 * (those are Clang-only), and these tiny loops vectorize/unroll fine. */
static inline float vmin(vf v)
{
	float m = v[0];
	for (int k = 1; k < KD_W; k++) if (v[k] < m) m = v[k];
	return m;
}
static inline int any_negative(vf v)
{
	for (int k = 0; k < KD_W; k++) if (v[k] < 0.0f) return 1;
	return 0;
}

static void nn_search_simd(const KDTreeV *t, vf qx, vf qy, vf qz,
                   vi *best_idx, vf *best_d2)
{
	int stack[64];
	vf bound[64];
	int sp = 0;
	stack[sp] = t->root; bound[sp] = (vf){0}; sp++;

	vf bd = *best_d2;
	vi bi = *best_idx;

	while (sp > 0)
	{
		sp--;
		/* skip only if NO lane can beat its plane bound */
		if (vmin(bound[sp] - bd) >= 0.0f) continue;
		const KDNodeV *nd = &t->nodes[stack[sp]];

		/* leaf: scalar loop over bucket points, SIMD across queries */
		if (nd->count >= 0)
		{
			for (int p = 0; p < nd->count; p++)
			{
				vf dx = nd->xs[p] - qx;	/* broadcast point p */
				vf dy = nd->ys[p] - qy;
				vf dz = nd->zs[p] - qz;
				vf d2 = (dx * dx) + (dy * dy) + (dz * dz);

				vi mask = (d2 < bd);
				vi pidx;
				for (int k = 0; k < KD_W; k++)
				{
					pidx[k] = nd->idx[p];
				}
				bi = (mask & pidx) | (~mask & bi);
				bd = (vf)((mask & (vi)d2) | (~mask & (vi)bd));
			}
			continue;
		}

		/* internal */
		int   ax   = nd->axis;
		vf q    = (ax == 0) ? qx : (ax == 1) ? qy : qz;
		vf diff = q - nd->split;
		vf pd2 = diff*diff;	/* per-lane plane distance squared */

		/* shared near/far order: go where any lane wants to go */
		int goleft = any_negative(diff);
		int near = goleft ? nd->left  : nd->right;
		int far  = goleft ? nd->right : nd->left;

		stack[sp] = far;
		bound[sp] = pd2;
		sp++;
		stack[sp] = near;
		bound[sp] = (vf){0};
		sp++;
	}
	*best_d2 = bd;
	*best_idx = bi;
}

void kd_nearest_simd(const KDTreeV *t, vf qx, vf qy, vf qz,
                vi *best_idx, vf *best_d2)
{
	for (int k = 0; k < KD_W; k++)
	{
		(*best_idx)[k] = -1;
		(*best_d2)[k]  = FLT_MAX;
	}
	nn_search_simd(t, qx, qy, qz, best_idx, best_d2);
}

void bf_nearest_simd(const PointCloud *pts, vf qx, vf qy, vf	qz,
                vi *best_idx, vf *best_d2)
{
	vf bd;
	vi bi;
	for (int k = 0; k < KD_W; k++) { bd[k] = FLT_MAX; bi[k] = -1; }

	for (int i = 0; i < pts->n; i++) {
		float xs = pts->x[i];
		float ys = pts->y[i];
		float zs = pts->z[i];
		vf dx = xs - qx;	/* broadcasting */	
		vf dy = ys - qy;
		vf dz = zs - qz;
		vf d2 = (dx * dx) + (dy * dy) + (dz * dz);

		vi mask = (d2 < bd);
		bd = (vf)((mask & (vi)d2) | (~mask & (vi)bd));
		vi ib;
		for(int k = 0; k < KD_W; k++)
		{
			ib[k] = i;
		}
		bi = (mask & ib) | (~mask & bi);
	}

	*best_idx = bi;
	*best_d2  = bd;
}
