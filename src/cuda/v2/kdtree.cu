#include "kdtree_cuda.h" 

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

static inline float coord(const PointCloud *p, int i, int axis)
{
	return axis == 0 ? p->x[i] : axis == 1 ? p->y[i] : p->z[i];
}

static inline void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

static int partition(KDTreeGPU *t, int lo, int hi, int axis)
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
static void qselect(KDTreeGPU *t, int lo, int hi, int k, int axis)
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
static int build_rec(KDTreeGPU *t, int *next, int lo, int hi, int depth)
{
	if ((hi - lo) <= KD_W)	/* leaf: 1..KD_W points */
	{
		int node = (*next)++;
		int count = hi - lo;
		KDNodeGPU *nd = &t->nodes[node];

		/* Store this leaf's points contiguously in the side arrays at the
		 * leaf's perm offset. The leaf ranges [lo,hi) tile [0,n) exactly, so
		 * leaf_start = lo gives each leaf a disjoint, non-overlapping slot and
		 * no padding is needed -- the traversal scans exactly `count` points. */
		for (int i = 0; i < count; i++)
		{
			int p = t->perm[lo + i];
			t->leaf_x[lo + i]   = t->pts->x[p];
			t->leaf_y[lo + i]   = t->pts->y[p];
			t->leaf_z[lo + i]   = t->pts->z[p];
			t->leaf_idx[lo + i] = p;
		}
		nd->count = count;
		nd->leaf_start = lo;
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

static void flatten_bfs(KDTreeGPU *t, int n_nodes)
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

	KDNodeGPU *neu = (KDNodeGPU *)malloc((size_t)n_nodes * sizeof *neu);
	if (!neu) { perror("flatten_bfs"); exit(1); }
	for (int i = 0; i < n_nodes; i++)
	{
		KDNodeGPU nd = t->nodes[queue[i]];
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

void kd_build(KDTreeGPU *t, const PointCloud *pts)
{
	t->pts = pts;
	t->n   = pts->n;
	t->n_nodes = 0;
	t->root = -1;
	t->perm  = NULL;
	t->nodes = NULL;
	t->leaf_x = t->leaf_y = t->leaf_z = NULL;
	t->leaf_idx = NULL;
	if (pts->n == 0) return;

	t->perm  = (int *)malloc((size_t)pts->n * sizeof *t->perm);
	t->nodes = (KDNodeGPU *)malloc((size_t)pts->n * sizeof *t->nodes);  /* over-allocation: total nodes < n */
	/* leaf side arrays: one entry per point, indexed by leaf_start (= perm offset) */
	t->leaf_x   = (float *)malloc((size_t)pts->n * sizeof *t->leaf_x);
	t->leaf_y   = (float *)malloc((size_t)pts->n * sizeof *t->leaf_y);
	t->leaf_z   = (float *)malloc((size_t)pts->n * sizeof *t->leaf_z);
	t->leaf_idx = (int *)malloc((size_t)pts->n * sizeof *t->leaf_idx);
	if (!t->perm || !t->nodes || !t->leaf_x || !t->leaf_y || !t->leaf_z || !t->leaf_idx) {
		perror("kd_build"); exit(1);
	}
	for (int i = 0; i < pts->n; i++) t->perm[i] = i;

	int next = 0;
	t->root = build_rec(t, &next, 0, pts->n, 0);
	flatten_bfs(t, next);
	t->n_nodes = next;   /* flatten preserves the node count */
}

void kd_free(KDTreeGPU *t)
{
	free(t->perm);
	free(t->nodes);
	free(t->leaf_x);
	free(t->leaf_y);
	free(t->leaf_z);
	free(t->leaf_idx);
	t->perm = NULL;
	t->nodes = NULL;
	t->leaf_x = t->leaf_y = t->leaf_z = NULL;
	t->leaf_idx = NULL;
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
__device__ static void nn_search(const KDNodeGPU * __restrict__ nodes, int root,
				 const float * __restrict__ leaf_x,
				 const float * __restrict__ leaf_y,
				 const float * __restrict__ leaf_z,
				 const int   * __restrict__ leaf_idx,
				 float qxv, float qyv, float qzv,
                                 int   * __restrict__ best_idx,
				 float * __restrict__ best_d2)
{
	int   stack[32];
	float bound[32];
	int   sp = 0;
	stack[sp] = root; bound[sp] = 0.0f; sp++;

	float bd = FLT_MAX;
	int   bi = -1;

	while (sp > 0)
	{
		sp--;
		if (bound[sp] >= bd) continue;	/* pruning */
		const KDNodeGPU * __restrict__ nd = &nodes[stack[sp]];

		if (nd->count >= 0)	/* leaf: scan the bucket */
		{
			int base = nd->leaf_start;
			for (int i = 0; i < nd->count; i++)
			{
				float dx = leaf_x[base + i] - qxv;
				float dy = leaf_y[base + i] - qyv;
				float dz = leaf_z[base + i] - qzv;
				float d2 = (dx * dx) + (dy * dy) + (dz * dz);
				if (d2 < bd) { bd = d2; bi = leaf_idx[base + i]; }
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
	*best_d2  = bd;
	*best_idx = bi;
}

__global__ void kd_nearest_kernel(const KDNodeGPU * __restrict__ nodes, int root, int qn,
				  const float * __restrict__ leaf_x,
				  const float * __restrict__ leaf_y,
				  const float * __restrict__ leaf_z,
				  const int   * __restrict__ leaf_idx,
                                  const float * __restrict__ qx,
				  const float * __restrict__ qy,
				  const float * __restrict__ qz,
                                  int         * __restrict__ best_idx,
				  float       * __restrict__ best_d2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= qn) return;

	nn_search(nodes, root, leaf_x, leaf_y, leaf_z, leaf_idx, qx[idx], qy[idx], qz[idx], &best_idx[idx], &best_d2[idx]);
}

__global__ void bf_nearest_kernel(const float * __restrict__ tx, 
				  const float * __restrict__ ty, 
				  const float * __restrict__ tz, int tn,
                                  const float * __restrict__ qx, 
				  const float * __restrict__ qy, 
				  const float * __restrict__ qz, int qn,
                                  int         * __restrict__ best_idx, 
				  float       * __restrict__ best_d2)
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
