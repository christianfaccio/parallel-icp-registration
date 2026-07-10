#ifndef ICP_CUDA_KERNELS_V1_H_
#define ICP_CUDA_KERNELS_V1_H_

/*
 * v1 CUDA NN kernels. Like v0, this version reuses the CPU KDNodeV / KDTreeV
 * layout (fat leaf buckets, KD_W = 16) from kdtreeV.h, so the kernels take a
 * KDNodeV node pool.
 *
 * Kept local to src/cuda/v1/ on purpose -- a "#include" (quotes) resolves to
 * this directory before -Iinclude, so v1 gets its own tree layout while v2's
 * compact KDNodeGPU and v3's voxel grid keep theirs. Do NOT pull in
 * include/kdtree_cuda.h here: that header describes the v2 layout and would
 * redefine KD_W and clash with these signatures.
 */

#include "kdtreeV.h"   /* KDNodeV, KDTreeV, KD_W (=16), kd_build/kd_free/kd_nearest */

__global__ void kd_nearest_kernel(const KDNodeV *nodes, int root, int qn,
                                  const float *qx, const float *qy, const float *qz,
                                  int *best_idx, float *best_d2);

__global__ void bf_nearest_kernel(const float *tx, const float *ty, const float *tz, int tn,
                                  const float *qx, const float *qy, const float *qz, int qn,
                                  int *best_idx, float *best_d2);

#endif /* ICP_CUDA_KERNELS_V1_H_ */
