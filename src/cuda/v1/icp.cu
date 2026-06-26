/*
 * icp.c -- the point-to-point ICP loop. Each stage is written as a clean,
 * self-contained pass so the parallel backends can replace one at a time.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L   /* clock_gettime; nvcc already sets 200809L */
#endif

#include "icp.h"
#include "kdtreeV.h"
#include "kdtree_cuda.h"
#include "linalg.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)
	do {
		cudaError_t err_ = (call);
		if (err_ != cudaSuccess) {
			fprintf(stderr, "CUDA error %s:%d: %s\n",
			        __FILE__, __LINE__, cudaGetErrorString(err_));
			exit(1);
		}
	} while (0)

__global__ void reduce(const float *xs, const float *ys, const float *zs, 	 
		       const float *xt, const float *yt, const float *zt, 
		       const int n,
		       const int *match, double *cs, double *ct, int *m)
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
	atomicAdd(&m, (int)mask);
}

__global__ void compute_match(int *match, float *bd2, int *bi, float max_d2, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	match[idx] = (bd2[i] <= max_d2) ? bi[i] : -1;
}

__global__ void average(double *cs, double *ct, double m, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	cs[idx] /= (double)m; 
	ct[idx] /= (double)m; 
}

__global__ void compute_cc(const float *xs, const float *ys, const float *zs, 
		      const float *xt, const float *yt, const float *zt,
		      const int n, 
		      const int *match, double *cs, double *ct, long m,
		      double *H, double *sse)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	double mask = (double)(match[idx] >= 0);
	int j = (match[idx] >= 0) ? match[idx] : 0;
	double sx = xs[idx] - cs[0], sy = ys[idx] - cs[1], sz = zs[idx] - cs[2];
	double tx = xt[j] - ct[0], ty = yt[j] - ct[1], tz = zt[j] - ct[2];
	atomicAdd(&H[0], sx*tx*mask); 
	atomicAdd(&H[1], sx*ty*mask); 
	atomicAdd(&H[2], sx*tz*mask);
	atomicAdd(&H[3], sy*tx*mask); 
	atomicAdd(&H[4], sy*ty*mask); 
	atomicAdd(&H[5], sy*tz*mask);
	atomicAdd(&H[6], sz*tx*mask); 
	atomicAdd(&H[7], sz*ty*mask); 
	atomicAdd(&H[8], sz*tz*mask);
	double ex = cur.x[idx] - tgt->x[j];
	double ey = cur.y[idx] - tgt->y[j];
	double ez = cur.z[idx] - tgt->z[j];
	atomicAdd(&sse, (ex*ex + ey*ey + ez*ez)*mask);       /* residual at start of iter */
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

	/* --- device setup: done once, outside the iteration loop ------------ */
	int Ntpb = 256;
	int Nb   = (cur.n + Ntpb - 1) / Ntpb;

	float *d_qx, *d_qy, *d_qz;   /* per-iter query upload (cur moves each iter) */
	int   *d_bi;  float *d_bd2;  /* per-query results, written by the kernel    */
	CUDA_CHECK(cudaMalloc(&d_qx, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_qy, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_qz, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_bi,  (size_t)cur.n * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_bd2, (size_t)cur.n * sizeof(float)));

	int   *h_bi  = (int *)  malloc((size_t)cur.n * sizeof *h_bi);
	float *h_bd2 = (float *)malloc((size_t)cur.n * sizeof *h_bd2);

	/* backend-specific, uploaded once: the target never moves */
	KDNodeV *d_nodes = NULL;
	float   *d_tx = NULL, *d_ty = NULL, *d_tz = NULL;
	if (prm->use_kdtree) {
       		CUDA_CHECK(cudaMalloc(&d_nodes, (size_t)tree.n_nodes * sizeof(KDNodeV)));
		CUDA_CHECK(cudaMemcpy(d_nodes, tree.nodes,
				      (size_t)tree.n_nodes * sizeof(KDNodeV),
		      		      cudaMemcpyHostToDevice));		      
	} else {
		CUDA_CHECK(cudaMalloc(&d_tx, (size_t)tgt->n * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_ty, (size_t)tgt->n * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_tz, (size_t)tgt->n * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(d_tx, tgt->x, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_ty, tgt->y, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_tz, tgt->z, (size_t)tgt->n * sizeof(float), cudaMemcpyHostToDevice));
	}

	double Rtot[9], Ttot[3];
	mat3_identity(Rtot);
	Ttot[0] = Ttot[1] = Ttot[2] = 0.0;

	int   *match  = (int *)malloc((size_t)cur.n * sizeof *match);  /* tgt idx or -1 */
	CUDA_CHECK(cudaMalloc(&d_match, (size_t)cur.n * sizeof(&match)));
	double max_d2 = prm->max_corr_dist * prm->max_corr_dist;

	res->t_nn = res->t_solve = res->t_transform = 0.0;
	res->final_pairs = 0;

	double prev_rmse = HUGE_VAL;
	int it = 0;
	CUDA_CHECK(cudaMalloc(&d_xs, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_ys, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_zs, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_xt, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_yt, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_zt, (size_t)cur.n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_cs, 3 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_ct, 3 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_m, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&mask, sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_H, 9 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_sse, sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_Rs, 9 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_Ts, 3 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_Rn, 9 * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_Tn, 3 * sizeof(double)));
	for (; it < prm->max_iters; it++) {

		/* --- 1+2. nearest neighbour + rejection ------------------------- */
		double t0 = now_sec();
		
		CUDA_CHECK(cudaMemcpy(d_qx, cur.x, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_qy, cur.y, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_qz, cur.z, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		
		if (prm->use_kdtree)
			kd_nearest_kernel<<<Nb, Ntpb>>>(d_nodes, tree.root, cur.n,
			                                d_qx, d_qy, d_qz, d_bi, d_bd2);
		else
			bf_nearest_kernel<<<Nb, Ntpb>>>(d_tx, d_ty, d_tz, tgt->n,
			                                d_qx, d_qy, d_qz, cur.n, d_bi, d_bd2);
		CUDA_CHECK(cudaGetLastError());   /* catch a bad launch config */
		
		compute_match<<<Nb, Ntpb>>>(d_match, d_bd2, d_bi, max_d2, cur.n); 

		/* --- 3. centroids + cross-covariance H (reductions) ------------- */
		t0 = now_sec();
		CUDA_CHECK(cudaMemcpy(d_xs, cur.x, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_ys, cur.y, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_zs, cur.z, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_xt, tgt.x, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_yt, tgt.y, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_zt, tgt.z, (size_t)cur.n * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(d_cs, 0, 3 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_ct, 0, 3 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_m, 0, sizeof(int)));
		int m;
		double sse;

		CUDA_CHECK(cudaMemcpy()

		compute_centroids<<<Nb, Ntpb>>>(d_xs, d_ys, d_zs, d_xt, d_yt, d_zt, cur.n, match, cs, ct, m);

		cudaMemcpy(&m, d_m, sizeof(long), cudaMemcpyDeviceToHost);
		if (m < 3) { res->t_solve += now_sec() - t0; break; }  /* under-constrained */
		
		average<<<1, 3>>>(d_cs, d_ct, &d_m, cur.n);

		CUDA_CHECK(cudaMemset(d_H, 0, 9 * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_sse, 0, sizeof(double)));
		compute_cc<<<Nb, Ntpb>>>(d_xs, d_ys, d_zs, d_xt, d_yt, d_zt, cur.n, 
			match, d_cs, d_ct, &d_m, d_H, d_sse);
		CUDA_CHECK(cudaMemcpy(&sse, d_sse, sizeof(double), cudaMemcpyHostToDevice));

		/* --- 4a. solve for this step's R,T ------------------------------ */
		double Rs[9], Ts[3];
		double H, cs, ct;
		CUDA_CHECK(cudaMemcpy(&H, d_H, 9 * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&cs, d_cs, 9 * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(&ct, d_ct, 3 * sizeof(double), cudaMemcpyHostToDevice));
		kabsch_from_H(H, cs, ct, Rs, Ts);	// better to run on host 
		res->t_solve += now_sec() - t0;

		/* --- 4b. apply + accumulate the total transform ----------------- */
		t0 = now_sec();
		CUDA_CHECK(cudaMemcpy(d_Rs, Rs, 9 * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Ts, Ts, 3 * sizeof(double), cudaMemcpyHostToDevice));
		pc_apply_transform_k<<<Nb, Ntpb>>>(d_xs, d_ys, d_zs, cur.n, d_Rs, d_Ts);
		double Rn[9], Tn[3];
		mat3_mul(Rs, Rtot, Rn);                 /* Rtot <- Rs * Rtot */
		for (int k = 0; k < 9; k++) Rtot[k] = Rn[k];
		mat3_mul_vec(Rs, Ttot, Tn);             /* Ttot <- Rs * Ttot + Ts */
		Ttot[0] = Tn[0] + Ts[0];
		Ttot[1] = Tn[1] + Ts[1];
		Ttot[2] = Tn[2] + Ts[2];
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

	cudaFree(d_cur);
	cudaFree(d_bi); cudaFree(d_bd2);
	if (prm->use_kdtree) cudaFree(d_nodes);
	else { cudaFree(d_tx); cudaFree(d_ty); cudaFree(d_tz); }
	free(h_bi); free(h_bd2);

	free(match);
	if (prm->use_kdtree) kd_free(&tree);
	pc_free(&cur);
	return 0;
}
