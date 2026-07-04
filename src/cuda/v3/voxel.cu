#include "voxel_cuda.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>

void voxel_build(VoxelGrid *vg, const PointCloud *pts, int bits, int iters)
{
	vg->n = pts->n;
	vg->bits = bits;
	int G = 1<<bits;
	vg->G = G;
	vg->nv = (long)G*G*G;
	double bmin[3], bmax[3];
	pc_bounds(pts, bmin, bmax);
	for (int a = 0; a < 3; a++)
	{
		vg->origin[a] = bmin[a];
		double ext = bmax[a] - bmin[a];
		double cell = ext > 0 ? ext / G : 1.0;
		vg->inv_cell[a] = 1.0 / cell;
	}

	vg->offsets = malloc((size_t)(vg->nv + 1) * sizeof *vg->offsets);
	vg->x = malloc((size_t)vg->n * sizeof *vg->x);
	vg->y = malloc((size_t)vg->n * sizeof *vg->y);
	vg->z = malloc((size_t)vg->n * sizeof *vg->z);
	vg->idx = malloc((size_t)vg->n * sizeof *vg->idx);
	vg->voxel_root = malloc((size_t)vg->nv * sizeof *vg->voxel_root);
	
	int *count = calloc((size_t)vg->nv, sizeof *count);
	int *v = malloc((size_t)vg->n * sizeof *v);

	// quantization
	for (int i = 0; i < pts->n; i++)
	{
		int vx = (int)floor((pts->x[i] - vg->origin[0] * vg->inv_cell[0]);
		int vy = (int)floor((pts->y[i] - vg->origin[1] * vg->inv_cell[1]);
		int vz = (int)floor((pts->z[i] - vg->origin[2] * vg->inv_cell[2]);
		if (vx < 0) vx = 0; else if (vx > G - 1) vx = G - 1;
		if (vy < 0) vy = 0; else if (vy > G - 1) vy = G - 1;
		if (vz < 0) vz = 0; else if (vz > G - 1) vz = G - 1;

		v[i] = (vx * G + vy) * G + vz;
		count[v[i]]++;
	}
	
	vg->offsets[0] = 0;
	for (long c = 0; c < vg->nv; c++)
	{
		vg->offsets[c+1] = vg->offsets[c] + count[c];
	}

	int *cursor = malloc((size_t)vg->nv * sizeof *cursor);
	for (long c = 0; c < vg->nv; c++) cursor[c] = vg->offsets[c];
	for (int i = 0; i < vg->n; i++)
	{
		int slot = cursor[v[i]]++;
		vg->x[slot] = pts->x[i];
		vg->y[slot] = pts->y[i];
		vg->z[slot] = pts->z[i];
		vg->idx[slot] = i;
	}

	// Dilation

	for (long c = 0; c < vg->nv; c++)
		vg->voxel_root[c] = (vg->offsets[c+1] > vg->offsets[c]) ? (int)c : -1;
	
	long *frontier = malloc((size_t)vg->nv * sizeof *frontier);
	long *next = malloc((size_t)vg->nv * sizeof *next);
	long fn = 0;
	for (long c = 0; c < vg->nv; c++)
		if (vg->voxel_root[c] >= 0) frontier[fn++] = c;

	for (int iter = 0; iter < iters && fn > 0; iter++)
	{
		long nn = 0;
		for (long k = 0; k < fn; k++)
		{
			long f = frontier[k];
			int root = vg->voxel_root[f];
			int vz = f % G, vy = (f / G) % G, vx = f / ((long)G * G);

			#define TRY(nb) do {long _n=(nb); \
				if (vg->voxel_root[_n] == -1) { vg->voxel_root[_n] = root; next[nn++] = _n; } \
			} while (0)
			if (vz + 1 < G) TRY(f + 1);
			if (vz - 1 >= 0) TRY(f - 1);
			if (vy + 1 < G) TRY(f + G);
			if (vy - 1 >= 0) TRY(f - G);
			if (vx + 1 < G) TRY(f + (long)G*G);
			if (vx - 1 >= 0) TRY(f - (long)G*G);
			#undef TRY
		}
		long *tmp = frontier; 
		frontier = next;
		next = tmp;
		fn = nn;
	}

	free(cursor);
	free(count);
	free(v);
	free(frontier);
	free(next);
}

void voxel_free(VoxelGrid *vg)
{
	free(vg->x);
	free(vg->y);
	free(vg->z);
	free(vg->offsets);
	free(vg->voxel_root);
	free(vg->idx);
	vg->x = NULL;
	vg->y = NULL;
	vg->z = NULL;
	vg->offsets = NULL;
	vg->voxel_root = NULL;
	vg->idx = NULL;
}

__global__ void voxel_nn_kernel(const int   * __restrict__ offsets,
                                const int   * __restrict__ voxel_root,
                                const float * __restrict__ vx,
                                const float * __restrict__ vy,
                                const float * __restrict__ vz,
                                const int   * __restrict__ vidx,
                                int G, long nv,
                                float ox, float oy, float oz,
                                float icx, float icy, float icz,
                                const float * __restrict__ tx,
                                const float * __restrict__ ty,
                                const float * __restrict__ tz, int tn,
                                const float * __restrict__ qx,
                                const float * __restrict__ qy,
                                const float * __restrict__ qz, int qn,
                                int   * __restrict__ best_idx,
                                float * __restrict__ best_d2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= qn) return;

	float qxv = qx[idx], qyv = qy[idx], qzv = qz[idx];
	
	int qvx = (int)floor((qxv - ox) * icx);
	int qvy = (int)floor((qyv - oy) * icy);
	int qvz = (int)floor((qzv - oz) * icz);
	if (qvx < 0) qvx = 0; else if (qvx > G - 1) qvx = G - 1;
	if (qvy < 0) qvy = 0; else if (qvy > G - 1) qvy = G - 1;
	if (qvz < 0) qvz = 0; else if (qvz > G - 1) qvz = G - 1;
	
	long qv = ((long)qvx * G + qvy) * G * qvz;
	int r = voxel_root[qv];
	
	float bd = FLT_MAX;
	int bi = -1;

	if (r < 0)
	{
		for (int i = 0; i < tn; i++)
		{
			float dx = tx[i] - qxv;
			float dy = ty[i] - qyv;
			float dz = tz[i] - qzv;
			float d2 = (dx * dx) + (dy * dy) + (dz * dz);
			if (d2 < bd) { bd = d2; bi = i; }
	} 
	else
	{
		int start = offsets[r], end = offsets[r+1];
		for (int i = start; i < end; i++)
		{
			float dx = vx[i] - qxv;
			float dy = vy[i] - qyv;
			float dz = vz[i] - qzv;
			float d2 = (dx * dx) + (dy * dy) + (dz * dz);
			if (d2 < bd) { bd = d2; bi = i; }
	}

	best_idx[idx] = bi;
	best_d2[idx] = bd;	
}
