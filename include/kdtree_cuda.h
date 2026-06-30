#ifndef ICP_CUDA_KERNELS_H_
#define ICP_CUDA_KERNELS_H_

#include "pointcloud.h"

/*
 * GPU leaf bucket size. Unlike the CPU backends (where KD_W is the SIMD width),
 * on the GPU the leaf is scanned by a single thread with a scalar loop, so this
 * is purely a tree-depth vs. bucket-scan tuning knob -- a small value (4-8)
 * keeps the tree shallow enough to limit warp divergence without over-scanning.
 */
#define KD_W 4

typedef struct {
	int count;
	int axis;
	float split;
	int left, right;
	int leaf_start;
} KDNodeGPU;

typedef struct {
	const PointCloud *pts;
	int *perm;
	KDNodeGPU *nodes;
	float *leaf_x, *leaf_y, *leaf_z;
	int *leaf_idx;
	int n;
	int n_nodes;
	int root;
} KDTreeGPU;

void kd_build(KDTreeGPU *t, const PointCloud *pts);
void kd_free(KDTreeGPU *t);

__global__ void kd_nearest_kernel(const KDNodeGPU * __restrict__ nodes, int root, int qn,
				  const float * __restrict__ leaf_x,
				  const float * __restrict__ leaf_y,
				  const float * __restrict__ leaf_z,
				  const int * __restrict__ leaf_idx,
                                  const float * __restrict__ qx, 
				  const float * __restrict__ qy,
				  const float * __restrict__ qz,
                                  int * __restrict__ best_idx,
				  float * __restrict__ best_d2);

__global__ void bf_nearest_kernel(const float * __restrict__ tx, 
				  const float * __restrict__ ty, 
				  const float * __restrict__ tz, int tn,
                                  const float * __restrict__ qx, 
				  const float * __restrict__ qy, 
				  const float * __restrict__ qz, int qn,
                                  int * __restrict__ best_idx, 
				  float * __restrict__ best_d2);

#endif /* ICP_CUDA_KERNELS_H_ */
