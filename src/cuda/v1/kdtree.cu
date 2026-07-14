#include "kdtree_cuda.h"   /* pulls in kdtreeV.h: KDNodeV, KDTreeV, KD_W */

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
	int *queue = (int *)malloc((size_t)n_nodes * sizeof *queue);
	int *newidx = (int *)malloc((size_t)n_nodes * sizeof *newidx);
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

	KDNodeV *neu = (KDNodeV *)malloc((size_t)n_nodes * sizeof *neu);
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
	t->n_nodes = 0;
	t->root = -1;
	t->perm  = NULL;
	t->nodes = NULL;
	if (pts->n == 0) return;

	t->perm  = (int *)malloc((size_t)pts->n * sizeof *t->perm);
	t->nodes = (KDNodeV *)malloc((size_t)pts->n * sizeof *t->nodes);  /* over-allocation: total nodes < n */
	if (!t->perm || !t->nodes) { perror("kd_build"); exit(1); }
	for (int i = 0; i < pts->n; i++) t->perm[i] = i;

	int next = 0;
	t->root = build_rec(t, &next, 0, pts->n, 0);
	flatten_bfs(t, next);
	t->n_nodes = next;   /* flatten preserves the node count */
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

/*
 * Per-thread scalar kd-tree traversal. Each GPU thread chases the (flattened)
 * tree for a single query (qxv,qyv,qzv) using a private depth-first stack; the
 * `bound` entry is the squared distance from the query to the splitting plane,
 * so a subtree is pruned when even its closest possible point can't beat the
 * current best. The KD_W-wide leaf buckets are scanned with an ordinary scalar
 * loop -- on the GPU the parallelism comes from the thread grid, not SIMD.
 */
__device__ static void nn_search(const KDNodeV *nodes, int root,
                                 float qxv, float qyv, float qzv,
                                 int *out_idx, float *out_d2)
{
	int   stack[64];
	float bound[64];
	int   sp = 0;
	stack[sp] = root; bound[sp] = 0.0f; sp++;

	float bd = FLT_MAX;
	int   bi = -1;

	while (sp > 0)
	{
		sp--;
		if (bound[sp] >= bd) continue;	/* pruning */
		const KDNodeV *nd = &nodes[stack[sp]];

		if (nd->count >= 0)	/* leaf: scan the bucket */
		{
			for (int i = 0; i < nd->count; i++)
			{
				float dx = nd->xs[i] - qxv;
				float dy = nd->ys[i] - qyv;
				float dz = nd->zs[i] - qzv;
				float d2 = (dx * dx) + (dy * dy) + (dz * dz);
				if (d2 < bd) { bd = d2; bi = nd->idx[i]; }
			}
			continue;
		}

		/* internal node */
		int   ax   = nd->axis;
		float q    = (ax == 0) ? qxv : (ax == 1) ? qyv : qzv;
		float diff = q - nd->split;

		int near = diff < 0 ? nd->left  : nd->right;
		int far  = diff < 0 ? nd->right : nd->left;

		stack[sp] = far;  bound[sp] = diff * diff; sp++;  /* visit far only if it can beat best */
		stack[sp] = near; bound[sp] = 0.0f;        sp++;  /* visit near first */
	}
	*out_d2  = bd;
	*out_idx = bi;
}

__global__ void kd_nearest_kernel(const KDNodeV *nodes, int root, int qn,
                                  const float *qx, const float *qy, const float *qz,
                                  int *best_idx, float *best_d2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= qn) return;

	nn_search(nodes, root, qx[idx], qy[idx], qz[idx], &best_idx[idx], &best_d2[idx]);
}

__global__ void bf_nearest_kernel(const float *tx, const float *ty, const float *tz, int tn,
                                  const float *qx, const float *qy, const float *qz, int qn,
                                  int *best_idx, float *best_d2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= qn) return;

	float qxv = qx[idx];
	float qyv = qy[idx];
	float qzv = qz[idx];

	float bd = FLT_MAX;
	int   bi = -1;

	for (int i = 0; i < tn; i++)
	{
		float dx = tx[i] - qxv;
		float dy = ty[i] - qyv;
		float dz = tz[i] - qzv;
		float d2 = (dx * dx) + (dy * dy) + (dz * dz);
		if (d2 < bd) { bd = d2; bi = i; }
	}
	best_idx[idx] = bi;
	best_d2[idx]  = bd;
}
