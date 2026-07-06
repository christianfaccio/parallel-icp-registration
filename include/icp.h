#ifndef ICP_ICP_H_
#define ICP_ICP_H_

/*
 * Point-to-point Iterative Closest Point (Kabsch / SVD solve).
 *
 * Each iteration runs the four-stage pipeline:
 *   1. nearest-neighbour search   (src point -> closest tgt point)
 *   2. correspondence rejection    (drop pairs farther than max_corr_dist)
 *   3. reduction                   (centroids + 3x3 cross-covariance H)
 *   4. solve + apply               (SVD/Kabsch -> R,T; transform src; accumulate)
 * looping until the RMSE stops improving or max_iters is reached.
 */

#include "pointcloud.h"

typedef struct {
	int    max_iters;       /* iteration cap */
	double tol;             /* stop when |delta RMSE| < tol */
	double max_corr_dist;   /* reject correspondences farther than this */
	int    use_kdtree;      /* 1 = KD-tree, 0 = brute force */
	int    use_voxel;       /* 1 = voxel grid (CUDA v3); ignored by other backends */
} ICPParams;

typedef struct {
	double R[9];            /* recovered rotation (row-major) */
	double T[3];            /* recovered translation */
	int    iters;           /* iterations actually run */
	double final_rmse;      /* RMSE of accepted correspondences at convergence */
	long   final_pairs;     /* number of accepted correspondences in last iter */
	double t_setup;         /* one-time: host tree build + Morton reorder      */
	double t_ctx;           /* one-time: CUDA context init (0 on CPU backends) */
	double t_upload;        /* one-time: device alloc + H2D upload (0 on CPU)  */
	double t_nn;            /* cumulative seconds: NN search          */
	double t_solve;         /* cumulative seconds: reduction + SVD     */
	double t_transform;     /* cumulative seconds: apply transform     */
} ICPResult;

/* Align `src` onto `tgt`. Inputs are not modified. Returns 0 on success. */
int icp_run(const PointCloud *src, const PointCloud *tgt,
            const ICPParams *prm, ICPResult *res);

#endif /* ICP_ICP_H_ */
