#include "voxel_cuda.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define CUDA_CHECK(call)                                                      \
	do {                                                                  \
		cudaError_t err_ = (call);                                    \
		if (err_ != cudaSuccess) {                                    \
			fprintf(stderr, "CUDA error %s:%d: %s\n",             \
			        __FILE__, __LINE__, cudaGetErrorString(err_));\
			exit(1);                                              \
		}                                                             \
	} while (0)

#define VOX_BLOCK 256

/* Quantize a point to its voxel and flatten to a linear index.
 * Layout: v = (vx*G + vy)*G + vz  (must match the decode in dilate/NN). */
__device__ __forceinline__ long voxel_of(float px, float py, float pz,
                                         int G,
                                         float ox, float oy, float oz,
                                         float icx, float icy, float icz)
{
	int vx = (int)floorf((px - ox) * icx);
	int vy = (int)floorf((py - oy) * icy);
	int vz = (int)floorf((pz - oz) * icz);
	if (vx < 0) vx = 0; else if (vx > G - 1) vx = G - 1;
	if (vy < 0) vy = 0; else if (vy > G - 1) vy = G - 1;
	if (vz < 0) vz = 0; else if (vz > G - 1) vz = G - 1;
	return ((long)vx * G + vy) * G + vz;
}

/* Pass 1: count how many points fall in each voxel. */
__global__ void hist_kernel(const float * __restrict__ px,
                            const float * __restrict__ py,
                            const float * __restrict__ pz,
                            int G, int n, int * __restrict__ count,
                            float ox, float oy, float oz,
                            float icx, float icy, float icz)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;

	long v = voxel_of(px[idx], py[idx], pz[idx], G,
	                  ox, oy, oz, icx, icy, icz);
	atomicAdd(&count[v], 1);
}

/* Pass 2: scatter each point into its voxel's slice of the sorted arrays.
 * `cursor` starts as a copy of `offsets`; the atomic hands out slots. */
__global__ void scatter_kernel(const float * __restrict__ px,
                               const float * __restrict__ py,
                               const float * __restrict__ pz,
                               float * __restrict__ vx,
                               float * __restrict__ vy,
                               float * __restrict__ vz,
                               int   * __restrict__ vidx,
                               int G, int n, int * __restrict__ cursor,
                               float ox, float oy, float oz,
                               float icx, float icy, float icz)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;

	long v = voxel_of(px[idx], py[idx], pz[idx], G,
	                  ox, oy, oz, icx, icy, icz);
	int slot = atomicAdd(&cursor[v], 1);
	vx[slot]   = px[idx];
	vy[slot]   = py[idx];
	vz[slot]   = pz[idx];
	vidx[slot] = idx;           /* original point index */
}

/* Seed the dilation: a voxel points to itself if non-empty, else -1. */
__global__ void init_root_kernel(const int * __restrict__ offsets,
                                 int * __restrict__ voxel_root, long nv)
{
	long idx = (long)threadIdx.x + (long)blockIdx.x * blockDim.x;
	if (idx >= nv) return;

	voxel_root[idx] = (offsets[idx + 1] > offsets[idx]) ? (int)idx : -1;
}

/* One dilation step (ping-pong: read `src`, write `dst`). An empty voxel
 * adopts the root of the first non-empty 6-neighbour it finds, so after a few
 * iterations every query voxel resolves to some nearby occupied voxel. */
__global__ void dilate_kernel(const int * __restrict__ src,
                              int * __restrict__ dst, int G, long nv)
{
	long idx = (long)threadIdx.x + (long)blockIdx.x * blockDim.x;
	if (idx >= nv) return;

	int r = src[idx];
	if (r >= 0) { dst[idx] = r; return; }   /* already occupied: keep */

	long GG = (long)G * G;
	long vz = idx % G;
	long vy = (idx / G) % G;
	long vx = idx / GG;

	int nr = -1;
	if      (vz + 1 < G  && src[idx + 1]  >= 0) nr = src[idx + 1];
	else if (vz - 1 >= 0 && src[idx - 1]  >= 0) nr = src[idx - 1];
	else if (vy + 1 < G  && src[idx + G]  >= 0) nr = src[idx + G];
	else if (vy - 1 >= 0 && src[idx - G]  >= 0) nr = src[idx - G];
	else if (vx + 1 < G  && src[idx + GG] >= 0) nr = src[idx + GG];
	else if (vx - 1 >= 0 && src[idx - GG] >= 0) nr = src[idx - GG];
	dst[idx] = nr;
}

void voxel_build(VoxelGrid *vg, const PointCloud *pts, int bits, int iters)
{
	int n = pts->n;
	vg->n = n;
	vg->bits = bits;
	int G = 1 << bits;
	vg->G = G;
	long nv = (long)G * G * G;
	vg->nv = nv;

	double bmin[3], bmax[3];
	pc_bounds(pts, bmin, bmax);
	float origin[3], inv_cell[3];
	for (int a = 0; a < 3; a++) {
		origin[a] = (float)bmin[a];
		vg->origin[a] = bmin[a];
		double ext = bmax[a] - bmin[a];
		double cell = ext > 0 ? ext / G : 1.0;
		inv_cell[a] = (float)(1.0 / cell);
		vg->inv_cell[a] = 1.0 / cell;
	}
	float ox = origin[0],  oy = origin[1],  oz = origin[2];
	float icx = inv_cell[0], icy = inv_cell[1], icz = inv_cell[2];

	/* host output buffers (voxel_free() releases these) */
	vg->x          = (float *)malloc((size_t)n * sizeof(float));
	vg->y          = (float *)malloc((size_t)n * sizeof(float));
	vg->z          = (float *)malloc((size_t)n * sizeof(float));
	vg->idx        = (int   *)malloc((size_t)n * sizeof(int));
	vg->offsets    = (int   *)malloc((size_t)(nv + 1) * sizeof(int));
	vg->voxel_root = (int   *)malloc((size_t)nv * sizeof(int));

	/* --- device buffers --- */
	float *d_pcx, *d_pcy, *d_pcz;                  /* input cloud */
	CUDA_CHECK(cudaMalloc(&d_pcx, (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_pcy, (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_pcz, (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_pcx, pts->x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_pcy, pts->y, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_pcz, pts->z, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

	float *d_x, *d_y, *d_z;                         /* voxel-sorted cloud */
	int   *d_idx;
	CUDA_CHECK(cudaMalloc(&d_x,   (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_y,   (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_z,   (size_t)n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_idx, (size_t)n * sizeof(int)));

	int *d_count, *d_offsets, *d_cursor;
	CUDA_CHECK(cudaMalloc(&d_count,   (size_t)nv * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_offsets, (size_t)(nv + 1) * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_cursor,  (size_t)nv * sizeof(int)));
	CUDA_CHECK(cudaMemset(d_count, 0, (size_t)nv * sizeof(int)));

	int *d_root, *d_root2;                          /* dilation ping-pong */
	CUDA_CHECK(cudaMalloc(&d_root,  (size_t)nv * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_root2, (size_t)nv * sizeof(int)));

	int gn  = (n + VOX_BLOCK - 1) / VOX_BLOCK;
	int gnv = (int)((nv + VOX_BLOCK - 1) / VOX_BLOCK);

	/* pass 1: histogram */
	hist_kernel<<<gn, VOX_BLOCK>>>(d_pcx, d_pcy, d_pcz, G, n, d_count,
	                               ox, oy, oz, icx, icy, icz);
	CUDA_CHECK(cudaGetLastError());

	/* exclusive prefix sum: count -> offsets[0..nv), then offsets[nv] = n */
	thrust::device_ptr<int> t_count(d_count), t_off(d_offsets);
	thrust::exclusive_scan(thrust::device, t_count, t_count + nv, t_off);
	CUDA_CHECK(cudaMemcpy(d_offsets + nv, &n, sizeof(int), cudaMemcpyHostToDevice));

	/* pass 2: scatter (cursor is a working copy of offsets) */
	CUDA_CHECK(cudaMemcpy(d_cursor, d_offsets, (size_t)nv * sizeof(int),
	                      cudaMemcpyDeviceToDevice));
	scatter_kernel<<<gn, VOX_BLOCK>>>(d_pcx, d_pcy, d_pcz,
	                                  d_x, d_y, d_z, d_idx,
	                                  G, n, d_cursor,
	                                  ox, oy, oz, icx, icy, icz);
	CUDA_CHECK(cudaGetLastError());

	/* dilation */
	init_root_kernel<<<gnv, VOX_BLOCK>>>(d_offsets, d_root, nv);
	CUDA_CHECK(cudaGetLastError());
	int *cur = d_root, *nxt = d_root2;
	for (int iter = 0; iter < iters; iter++) {
		dilate_kernel<<<gnv, VOX_BLOCK>>>(cur, nxt, G, nv);
		CUDA_CHECK(cudaGetLastError());
		int *tmp = cur; cur = nxt; nxt = tmp;
	}

	/* results back to host */
	CUDA_CHECK(cudaMemcpy(vg->offsets, d_offsets, (size_t)(nv + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(vg->x, d_x, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(vg->y, d_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(vg->z, d_z, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(vg->idx, d_idx, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(vg->voxel_root, cur, (size_t)nv * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(d_pcx); cudaFree(d_pcy); cudaFree(d_pcz);
	cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_idx);
	cudaFree(d_count); cudaFree(d_offsets); cudaFree(d_cursor);
	cudaFree(d_root); cudaFree(d_root2);
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

/* One thread per query: look up the query's (dilated) voxel and scan its
 * points; if the voxel resolves to nothing (-1), fall back to brute force.
 * best_idx is returned in the ORIGINAL target index space (via vidx). */
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

	int qvx = (int)floorf((qxv - ox) * icx);
	int qvy = (int)floorf((qyv - oy) * icy);
	int qvz = (int)floorf((qzv - oz) * icz);
	if (qvx < 0) qvx = 0; else if (qvx > G - 1) qvx = G - 1;
	if (qvy < 0) qvy = 0; else if (qvy > G - 1) qvy = G - 1;
	if (qvz < 0) qvz = 0; else if (qvz > G - 1) qvz = G - 1;

	long qv = ((long)qvx * G + qvy) * G + qvz;
	int r = voxel_root[qv];

	float bd = FLT_MAX;
	int bi = -1;

	if (r < 0) {
		for (int i = 0; i < tn; i++) {
			float dx = tx[i] - qxv;
			float dy = ty[i] - qyv;
			float dz = tz[i] - qzv;
			float d2 = dx * dx + dy * dy + dz * dz;
			if (d2 < bd) { bd = d2; bi = i; }
		}
	} else {
		int start = offsets[r], end = offsets[r + 1];
		for (int i = start; i < end; i++) {
			float dx = vx[i] - qxv;
			float dy = vy[i] - qyv;
			float dz = vz[i] - qzv;
			float d2 = dx * dx + dy * dy + dz * dz;
			if (d2 < bd) { bd = d2; bi = vidx[i]; }
		}
	}

	best_idx[idx] = bi;
	best_d2[idx]  = bd;
}
