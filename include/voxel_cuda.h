#ifndef VOXEL_CUDA_H_
#define VOXEL_CUDA_H

#include "pointcloud.h"

typedef struct {
	int n;			// target points
	int bits;		// number of bits per axis
	int G;			// cells per axis = 1<<bits = 1*2^bits
	long nv;		// total voxels = (long)G*G*G
	double origin[3];	// target bbox min
	double inv_cell[3];	// 1/cell_size per axis: v = floor((p-o)*inv)

	float *x;
	float *y;
	float *z;
	int *offsets;
	int *voxel_root;
	int *idx;
} VoxelGrid;

void voxel_build(VoxelGrid *vg, const PointCloud *pts, int bits, int iters);
void voxel_free(VoxelGrid *vg);


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
                                float * __restrict__ best_d2);

#endif /* VOXEL_CUDA_H */
