#ifndef ICP_CUDA_KERNELS_H_
#define ICP_CUDA_KERNELS_H_

/*
 * CUDA NN kernels. One thread per query: thread `idx` searches the target for
 * the point nearest to query `idx` and writes best_idx[idx] / best_d2[idx].
 * The host plumbing (device allocations, H2D query upload, D2H result copy)
 * lives in icp.cu; the kd-tree itself is built on the host (kdtree.cu) and the
 * flat node pool is shipped to the device with a single cudaMemcpy.
 *
 * CUDA-only header: include from .cu translation units (it uses __global__).
 */

#include "kdtreeV.h"   /* KDNodeV, KDTreeV, KD_W, PointCloud */

__global__ void kd_nearest_kernel(const KDNodeV *nodes, int root, int qn,
                                  const float *qx, const float *qy, const float *qz,
                                  int *best_idx, float *best_d2);

__global__ void bf_nearest_kernel(const float *tx, const float *ty, const float *tz, int tn,
                                  const float *qx, const float *qy, const float *qz, int qn,
                                  int *best_idx, float *best_d2);

#endif /* ICP_CUDA_KERNELS_H_ */
