#include "voxel_cuda.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>

// helper for quantization and v computation
__device__ __forceinline__ long voxel_of(float px, float py, float pz,
					 int G, 
					 float ox, float oy, float oz,
					 float icx, float icy, float icz)
{
	int vx = (int)floor((px - ox) * icx);
	int vy = (int)floor((py - oy) * icy);
	int vz = (int)floor((pz - oz) * icz);
	if (vx < 0) vx = 0; else if (vx > G - 1) vx = G - 1;
	if (vy < 0) vy = 0; else if (vy > G - 1) vy = G - 1; 
	if (vz < 0) vz = 0; else if (vz > G - 1) vz = G - 1;
	return ((long)vx * G + vy) * G + vz;
}

__global__ void hist_kernel(float __restrict__ *px, float __restrict__ *py, float __restrict__ *pz,
			    int G, int n, int *count,
			    float __restrict__ *ox, float __restrict__ *oy, float __restrict__ *oz,
			    float __restrict__ *icx, float __restrict__ *icy, float __restrict__ *icz)
{
	idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > n) return;

	long v = voxel_of(px[idx], py[idx], pz[idx], 
			  G, 
			  ox[idx], oy[idx], oz[idx], 
			  icx[idx], icy[idx], icz[idx]);
	atomicAdd(&count[v], 1);
}

__global__ scatter_kernel(float __restrict__ *px, float __restrict__ *py, float __restrict__ *pz,
			  float *vx, float *vy, float *vz,
			  int G, int n, int __restrict__ *cursor,
			  float __restrict__ *ox, float __restrict__ *oy, float __restrict__ *oz,
			  float __restrict__ *icx, float __restrict__ *icy, float __restrict__ *icz))
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > n) return;
	
	long v = voxel_of(px[idx], py[idx], pz[idx],
			  G, 
			  ox[idx], oy[idx], oz[idx],
			  icx[idx], icy[idx], icz[idx])
	int slot = atomicAdd(&cursor[v], 1);
	vx[slot] = px[idx];
	vy[slot] = py[idx];
	vz[slot] = pz[idx];
	vidx[slot] = idx;
}

__global__ void init_root_kernel(float *voxel_root, float __restrict__ offsets, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > n) return;

	voxel_root[idx] = (offsets[idx+1] > offsets[idx]) ? (int)idx : -1;
}

__global__ void offsets_kernel(int *offsets, int *count, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > n) return;

	offsets[0] = 0;
	offsets[idx+1] = offsets[idx] + count[idx];
}

__global__ void dilate_kernel(int *voxel_root, int G, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > n) return;
	
	// check if voxel is empty, otherwise exit
	if (voxel_root[idx] >= 0) 
	{
		return;
	}

	int vz = f % G, vy = (f / G) % G, vx = f / ((long)G * G); 	// decompose
	
	if (vz + 1 < G) { voxel_root[idx] = voxel_root[idx+1]; return; }
	if (vz - 1 >= 0) { voxel_root[idx] = voxel_root[idx-1]; return; }
	if (vy + 1 < G) { voxel_root[idx] = voxel_root[idx+G]; return; }
	if (vy - 1 >= 0) { voxel_root[idx] = voxel_root[idx-G]; return; }
	if (vx + 1 < G) { voxel_root[idx] = voxel_root[idx+G*G]; return; }
	if (vx - 1 >= 0) { voxel_root[idx] = voxel_root[idx-G*G]; return; }
}

void voxel_build(VoxelGrid *vg, const PointCloud *pts, int bits, int iters)
{
	int n = pts->n;
	vg->n = n;
	vg->bits = bits;
	int G = 1<<bits;
	vg->G = G;
	long nv = (long)G*G*G;
	vg->nv = nv;
	double bmin[3], bmax[3];
	pc_bounds(pts, bmin, bmax);
	float origin[3];
	float inv_cell[3];
	for (int a = 0; a < 3; a++)
	{
		origin[a] = bmin[a];
		vg->origin[a] = origin[a];
		double ext = bmax[a] - bmin[a];
		double cell = ext > 0 ? ext / G : 1.0;
		inv_cell[a] = 1.0 / cell;
		vg->inv_cell[a] = inv_cell[a];
	}

	int *d_offsets, *d_idx, *d_voxel_root; 
	float *d_x, *d_y, *d_z; 
	cudaMalloc(&d_offsets, (size_t)(nv + 1) * sizeof int);
	cudaMalloc(&d_x, (size_t)n * sizeof float);
	cudaMalloc(&d_y, (size_t)n * sizeof float);
	cudaMalloc(&d_z, (size_t)n * sizeof float);
	cudaMalloc(&d_idx, (size_t)n * sizeof int);
	cudaMalloc(&d_voxel_root, (size_t)nv * sizeof int);
	
	int *d_count;
	cudaMalloc(&d_count, (size_t)nv, sizeof int);
	cudaMemset(d_count, 0, (size_t)nv * sizeof(int));
	int *v;
	cudaMalloc(&v, (size_t)n * sizeof int);
	
	float *d_pcx, *d_pcy, *d_pcz;
	cudaMalloc(&d_pcx, (size_t)n * sizeof(float));
	cudaMalloc(&d_pcy, (size_t)n * sizeof(float));
	cudaMalloc(&d_pcz, (size_t)n * sizeof(float));
	cudaMemcpy(d_pcx, pts->x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pcy, pts->y, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pcz, pts->z, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

	int  *d_ox, *d_oy, *d_oz;
	cudaMalloc(&d_ox, sizeof(float));
	cudaMalloc(&d_oy, sizeof(float));
	cudaMalloc(&d_oz, sizeof(float));
	cudaMemcpy(d_ox, origin[0], sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_oy, origin[1], sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_oz, origin[2], sizeof(float), cudaMemcpyHostToDevice);

	int  *d_icx, *d_icy, *d_icz;
	cudaMalloc(&d_icx, sizeof(float));
	cudaMalloc(&d_icy, sizeof(float));
	cudaMalloc(&d_icz, sizeof(float));
	cudaMemcpy(d_icx, inv_cell[0], sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_icy, inv_cell[1], sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_icz, inv_cell[2], sizeof(float), cudaMemcpyHostToDevice);

	hist_kernel<<<1,n>>>(d_pcx, d_pcy, d_pcz,
			     G, n, d_count,
			     d_ox, d_oy, d_oz,
			     d_icx, d_icy, d_icz);
	
	offsets_kernel<<<1,nv>>>(d_count, d_offsets, nv);

	int *d_cursor;
	cudaMalloc(&d_cursor, (size_t)nv * sizeof int);
	cudaMemcpy(d_cursor, d_offsets, (size_t)nv * sizeof int, cudaMemcpyHostToDevice);

	scatter_kernel<<<1,n>>>(d_pcx, d_pcy, d_pcz,
				d_x, d_y, d_z,
			        G, n, d_cursor,
			        d_ox, d_oy, d_oz,
			        d_icx, d_icy, d_icz);	
	
	// Dilation
	init_root_kernel<<<1,nv>>>(d_offsets, d_voxel_root, nv);
	
	for (int iter = 0; iter < iters; iter++)
	{
		dilate_kernel<<<1,nv>>>(voxel_root, G, n);
	}

	vg->offsets = malloc((size_t)(vg->nv + 1) * sizeof *vg->offsets);
	vg->x = malloc((size_t)vg->n * sizeof *vg->x);
	vg->y = malloc((size_t)vg->n * sizeof *vg->y);
	vg->z = malloc((size_t)vg->n * sizeof *vg->z);
	vg->idx = malloc((size_t)vg->n * sizeof *vg->idx);
	vg->voxel_root = malloc((size_t)vg->nv * sizeof *vg->voxel_root);

	cudaMemcpy(d_offsets, vg->offsets, (size_t)(nv+1) * sizeof(int));
	cudaMemcpy(d_x, vg->x, (size_t)n * sizeof(float));
	cudaMemcpy(d_y, vg->y, (size_t)n * sizeof(float));
	cudaMemcpy(d_z, vg->z, (size_t)n * sizeof(float));
	cudaMemcpy(d_idx, vg->idx, (size_t)n * sizeof(int));
	cudaMemcpy(d_voxel_root, vg->voxel_root, (size_t)n * sizeof(int));
	
	cudaFree(d_cursor);
	cudaFree(d_count);
	cudaFree(v);
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
