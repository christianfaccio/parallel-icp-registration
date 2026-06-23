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

#define CUDA_CHECK(call)                                                      \
	do {                                                                  \
		cudaError_t err_ = (call);                                    \
		if (err_ != cudaSuccess) {                                    \
			fprintf(stderr, "CUDA error %s:%d: %s\n",            \
			        __FILE__, __LINE__, cudaGetErrorString(err_));\
			exit(1);                                             \
		}                                                            \
	} while (0)

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
	double max_d2 = prm->max_corr_dist * prm->max_corr_dist;

	res->t_nn = res->t_solve = res->t_transform = 0.0;
	res->final_pairs = 0;

	double prev_rmse = HUGE_VAL;
	int it = 0;
	for (; it < prm->max_iters; it++) {

		/* --- 1+2. nearest neighbour + rejection ------------------------- */
		double t0 = now_sec();
		/* cur was transformed in place last iter, so re-upload the queries */
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

		/* D2H copies block until the kernel finishes, so no explicit sync */
		CUDA_CHECK(cudaMemcpy(h_bi,  d_bi,  (size_t)cur.n * sizeof(int),   cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_bd2, d_bd2, (size_t)cur.n * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < cur.n; i++)
			match[i] = (h_bd2[i] <= max_d2) ? h_bi[i] : -1;
		res->t_nn += now_sec() - t0;

		/* --- 3. centroids + cross-covariance H (reductions) ------------- */
		t0 = now_sec();
		double cs[3] = {0,0,0}, ct[3] = {0,0,0};
		long   m = 0;
		double mask;
		for (int i = 0; i < cur.n; i++) {
			mask = (double)(match[i] >= 0);
			int j = (match[i] >= 0) ? match[i] : 0;	// safe
			cs[0] += cur.x[i] * mask; cs[1] += cur.y[i] * mask; cs[2] += cur.z[i] * mask;
			ct[0] += tgt->x[j] * mask; ct[1] += tgt->y[j] * mask; ct[2] += tgt->z[j] * mask;
			m += (long)mask;
		}
		if (m < 3) { res->t_solve += now_sec() - t0; break; }  /* under-constrained */
		for (int k = 0; k < 3; k++) { cs[k] /= (double)m; ct[k] /= (double)m; }

		double H[9] = {0,0,0,0,0,0,0,0,0};
		double sse  = 0.0;
		for (int i = 0; i < cur.n; i++) {
			mask = (double)(match[i] >= 0);
			int j = (match[i] >= 0) ? match[i] : 0;	// safe
			double sx = cur.x[i] - cs[0], sy = cur.y[i] - cs[1], sz = cur.z[i] - cs[2];
			double tx = tgt->x[j] - ct[0], ty = tgt->y[j] - ct[1], tz = tgt->z[j] - ct[2];
			H[0] += sx*tx*mask; H[1] += sx*ty*mask; H[2] += sx*tz*mask;
			H[3] += sy*tx*mask; H[4] += sy*ty*mask; H[5] += sy*tz*mask;
			H[6] += sz*tx*mask; H[7] += sz*ty*mask; H[8] += sz*tz*mask;
			double ex = cur.x[i] - tgt->x[j];
			double ey = cur.y[i] - tgt->y[j];
			double ez = cur.z[i] - tgt->z[j];
			sse += (ex*ex + ey*ey + ez*ez)*mask;       /* residual at start of iter */
		}

		/* --- 4a. solve for this step's R,T ------------------------------ */
		double Rs[9], Ts[3];
		kabsch_from_H(H, cs, ct, Rs, Ts);
		res->t_solve += now_sec() - t0;

		/* --- 4b. apply + accumulate the total transform ----------------- */
		t0 = now_sec();
		pc_apply_transform(&cur, Rs, Ts);
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

	cudaFree(d_qx); cudaFree(d_qy); cudaFree(d_qz);
	cudaFree(d_bi); cudaFree(d_bd2);
	if (prm->use_kdtree) cudaFree(d_nodes);
	else { cudaFree(d_tx); cudaFree(d_ty); cudaFree(d_tz); }
	free(h_bi); free(h_bd2);

	free(match);
	if (prm->use_kdtree) kd_free(&tree);
	pc_free(&cur);
	return 0;
}
