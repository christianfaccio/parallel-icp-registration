#ifndef ICP_CUDA_KERNELS_V1_H_
#define ICP_CUDA_KERNELS_V1_H_

#include "kdtreeV.h"   /* KDNodeV, KDTreeV, KD_W (=16), kd_build/kd_free/kd_nearest */

__global__ void kd_nearest_kernel(const KDNodeV *nodes, int root, int qn,
                                  const float *qx, const float *qy, const float *qz,
                                  int *best_idx, float *best_d2);

__global__ void bf_nearest_kernel(const float *tx, const float *ty, const float *tz, int tn,
                                  const float *qx, const float *qy, const float *qz, int qn,
                                  int *best_idx, float *best_d2);

#endif /* ICP_CUDA_KERNELS_V1_H_ */
