/*
 * icp.cu (v1) -- the point-to-point ICP loop, with the *whole* per-iteration
 * pipeline on the device. The working source cloud lives on the GPU and is
 * transformed in place each iteration, so the only host<->device traffic per
 * iteration is a handful of scalars (R,T down; H, centroids, sse, m up). The
 * 3x3 Kabsch/SVD solve stays on the host: it is 9 numbers with no parallelism,
 * and we already pay one sync per iteration for the convergence test.
 *
 * Stages per iteration:
 *   1+2. NN search + rejection   -> d_match
 *   3.   centroids + cross-cov H  -> d_cs, d_ct, d_H, d_sse   (atomic reductions)
 *   4a.  Kabsch solve (host)      -> Rs, Ts
 *   4b.  apply transform (device, in place) + accumulate (host)
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L   /* clock_gettime; nvcc already sets 200809L */
#endif

#include "icp.h"
#include "kdtreeV.h"
#include "kdtree_cuda.h"
#include "linalg.h"
#include "pointcloud.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
	do {                                                                  \
		cudaError_t err_ = (call);                                    \
		if (err_ != cudaSuccess) {                                    \
			fprintf(stderr, "CUDA error %s:%d: %s\n",             \
			        __FILE__, __LINE__, cudaGetErrorString(err_));\
			exit(1);                                              \
		}                                                             \
	} while (0)

/* --- correspondence rejection: match[i] = tgt idx, or -1 if too far ------- */
__global__ void compute_match(int *match, const float *bd2, const int *bi,
                              float max_d2, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	match[idx] = (bd2[idx] <= max_d2) ? bi[idx] : -1;
}

/* --- stage 3a: accumulate centroid sums and the matched-pair count -------- */
__global__ void compute_centroids(const float *xs, const float *ys, const float *zs,
                                  const float *xt, const float *yt, const float *zt,
                                  int n, const int *match,
                                  double *cs, double *ct, int *m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	double mask = (double)(match[idx] >= 0);
	int j = (match[idx] >= 0) ? match[idx] : 0;
	atomicAdd(&cs[0], xs[idx] * mask);
	atomicAdd(&cs[1], ys[idx] * mask);
	atomicAdd(&cs[2], zs[idx] * mask);
	atomicAdd(&ct[0], xt[j] * mask);
	atomicAdd(&ct[1], yt[j] * mask);
	atomicAdd(&ct[2], zt[j] * mask);
	atomicAdd(m, (int)mask);
}

/* --- stage 3b: turn the centroid sums into means (3 threads) -------------- */
__global__ void average(double *cs, double *ct, int m)
{
	int idx = threadIdx.x;          /* launched <<<1,3>>> */
	if (idx >= 3) return;
	cs[idx] /= (double)m;
	ct[idx] /= (double)m;
}

/* --- stage 3c: cross-covariance H and the start-of-iter SSE --------------- */
__global__ void compute_cc(const float *xs, const float *ys, const float *zs,
                           const float *xt, const float *yt, const float *zt,
                           int n, const int *match,
                           const double *cs, const double *ct,
                           double *H, double *sse)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	double mask = (double)(match[idx] >= 0);
	int j = (match[idx] >= 0) ? match[idx] : 0;
	double sx = xs[idx] - cs[0], sy = ys[idx] - cs[1], sz = zs[idx] - cs[2];
	double tx = xt[j]   - ct[0], ty = yt[j]   - ct[1], tz = zt[j]   - ct[2];
	atomicAdd(&H[0], sx*tx*mask);
	atomicAdd(&H[1], sx*ty*mask);
	atomicAdd(&H[2], sx*tz*mask);
	atomicAdd(&H[3], sy*tx*mask);
	atomicAdd(&H[4], sy*ty*mask);
	atomicAdd(&H[5], sy*tz*mask);
	atomicAdd(&H[6], sz*tx*mask);
	atomicAdd(&H[7], sz*ty*mask);
	atomicAdd(&H[8], sz*tz*mask);
	double ex = xs[idx] - xt[j];
	double ey = ys[idx] - yt[j];
	double ez = zs[idx] - zt[j];
	atomicAdd(sse, (ex*ex + ey*ey + ez*ez) * mask);   /* residual at start of iter */
}

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int icp_run(const PointCloud *src, const PointCloud *tgt,
            const ICPParams *prm, ICPResult *res)
{
	PointCloud cur;
	pc_morton_order(src, &cur);   /* working cloud: deep copy of src, Morton-ordered */

	KDTreeV tree;
	if (prm->use_kdtree) kd_build(&tree, tgt);

	int Ntpb = 256;
	int Nb   = (cur.n + Ntpb - 1) / Ntpb;

	/* --- device buffers: all allocated once, outside the loop ----------- */
	/* working source cloud: queries AND transform target, resident on device */
	float *d_sx, *d_sy, *d_sz;
	CUDA_CHECK(cudaMalloc(&d_sx, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_sy, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_sz, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_sx, cur.x, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_sy, cur.y, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_sz, cur.z, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));

	/* target coords: needed by the reduction (match[i] -> tgt point), uploaded once */
	float *d_tx, *d_ty, *d_tz;
	CUDA_CHECK(cudaMalloc(&d_tx, (size_t)tgt->n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_ty, (size_t)tgt->n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_tz, (size_t)tgt->n * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_tx, tgt->x, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_ty, tgt->y, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_tz, tgt->z, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));

	/* kd-tree node pool, uploaded once (the target never moves) */
	KDNodeV *d_nodes = NULL;
	if (prm->use_kdtree) {
		CUDA_CHECK(cudaMalloc(&d_nodes, (size_t)tree.n_nodes * sizeof(KDNodeV)));
		CUDA_CHECK(cudaMemcpy(d_nodes, tree.nodes,
		                      (size_t)tree.n_nodes * sizeof(KDNodeV),
		                      cudaMemcpyHostToDevice));
	}

	/* per-query NN results + match, and the small reduction outputs */
	int   *d_bi;    float *d_bd2;
	int   *d_match;
	double *d_cs, *d_ct, *d_H, *d_sse, *d_R, *d_T;
	int    *d_m;
	CUDA_CHECK(cudaMalloc(&d_bi,    (size_t)cur.n * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_bd2,   (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_match, (size_t)cur.n * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_cs,  3 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_ct,  3 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_m,       sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_H,   9 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_sse,     sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_R,   9 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_T,   3 * sizeof(double)));

	double Rtot[9], Ttot[3];
	mat3_identity(Rtot);
	Ttot[0] = Ttot[1] = Ttot[2] = 0.0;

	double max_d2 = prm->max_corr_dist * prm->max_corr_dist;

	res->t_nn = res->t_solve = res->t_transform = 0.0;
	res->final_pairs = 0;

	double prev_rmse = HUGE_VAL;
	int it = 0;
	for (; it < prm->max_iters; it++) {

		/* --- 1+2. nearest neighbour + rejection ------------------------- */
		double t0 = now_sec();
		if (prm->use_kdtree)
			kd_nearest_kernel<<<Nb, Ntpb>>>(d_nodes, tree.root, cur.n,
			                                d_sx, d_sy, d_sz, d_bi, d_bd2);
		else
			bf_nearest_kernel<<<Nb, Ntpb>>>(d_tx, d_ty, d_tz, tgt->n,
			                                d_sx, d_sy, d_sz, cur.n, d_bi, d_bd2);
		CUDA_CHECK(cudaGetLastError());
		compute_match<<<Nb, Ntpb>>>(d_match, d_bd2, d_bi, (float)max_d2, cur.n);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		res->t_nn += now_sec() - t0;

		/* --- 3. centroids + cross-covariance H (reductions) ------------- */
		t0 = now_sec();
		CUDA_CHECK(cudaMemset(d_cs, 0, 3 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_ct, 0, 3 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_m,  0,     sizeof(int)));
		compute_centroids<<<Nb, Ntpb>>>(d_sx, d_sy, d_sz, d_tx, d_ty, d_tz,
		                                cur.n, d_match, d_cs, d_ct, d_m);
		CUDA_CHECK(cudaGetLastError());

		int m;
		CUDA_CHECK(cudaMemcpy(&m, d_m, sizeof(int), cudaMemcpyDeviceToHost));
		if (m < 3) { res->t_solve += now_sec() - t0; break; }  /* under-constrained */

		average<<<1, 3>>>(d_cs, d_ct, m);
		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaMemset(d_H,   0, 9 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_sse, 0,     sizeof(double)));
		compute_cc<<<Nb, Ntpb>>>(d_sx, d_sy, d_sz, d_tx, d_ty, d_tz,
		                         cur.n, d_match, d_cs, d_ct, d_H, d_sse);
		CUDA_CHECK(cudaGetLastError());

		/* --- 4a. solve for this step's R,T on the host ------------------ */
		double H[9], cs[3], ct[3], sse;
		CUDA_CHECK(cudaMemcpy(H,    d_H,   9 * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(cs,   d_cs,  3 * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(ct,   d_ct,  3 * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&sse, d_sse,     sizeof(double), cudaMemcpyDeviceToHost));

		double Rs[9], Ts[3];
		kabsch_from_H(H, cs, ct, Rs, Ts);
		res->t_solve += now_sec() - t0;

		/* --- 4b. apply transform (device, in place) + accumulate (host) - */
		t0 = now_sec();
		CUDA_CHECK(cudaMemcpy(d_R, Rs, 9 * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_T, Ts, 3 * sizeof(double), cudaMemcpyHostToDevice));
		pc_transform_kernel<<<Nb, Ntpb>>>(d_sx, d_sy, d_sz, cur.n, d_R, d_T);
		CUDA_CHECK(cudaGetLastError());

		double Rn[9], Tn[3];
		mat3_mul(Rs, Rtot, Rn);                 /* Rtot <- Rs * Rtot */
		for (int k = 0; k < 9; k++) Rtot[k] = Rn[k];
		mat3_mul_vec(Rs, Ttot, Tn);             /* Ttot <- Rs * Ttot + Ts */
		Ttot[0] = Tn[0] + Ts[0];
		Ttot[1] = Tn[1] + Ts[1];
		Ttot[2] = Tn[2] + Ts[2];
		CUDA_CHECK(cudaDeviceSynchronize());
		res->t_transform += now_sec() - t0;

		double rmse = sqrt(sse / (double)m);
		res->final_rmse  = rmse;
		res->final_pairs = m;
		if (fabs(prev_rmse - rmse) < prm->tol) { it++; break; }
		prev_rmse = rmse;
	}

	res->iters = it;
	for (int k = 0; k < 9; k++) res->R[k] = Rtot[k];
	for (int k = 0; k < 3; k++) res->T[k] = Ttot[k];

	cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_sz);
	cudaFree(d_tx); cudaFree(d_ty); cudaFree(d_tz);
	if (prm->use_kdtree) cudaFree(d_nodes);
	cudaFree(d_bi); cudaFree(d_bd2); cudaFree(d_match);
	cudaFree(d_cs); cudaFree(d_ct); cudaFree(d_m);
	cudaFree(d_H);  cudaFree(d_sse);
	cudaFree(d_R);  cudaFree(d_T);

	if (prm->use_kdtree) kd_free(&tree);
	pc_free(&cur);
	return 0;
}
