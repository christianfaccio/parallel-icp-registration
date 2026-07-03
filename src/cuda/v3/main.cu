/*
 * main.c -- serial ICP driver and harness.
 *
 * Generates a synthetic target cloud, builds a source by applying a KNOWN
 * ground-truth transform (+ noise, partial overlap, outliers), runs ICP, and
 * reports convergence, accuracy vs ground truth, and a per-stage timing
 * breakdown (the data the parallelization phases will try to improve).
 *
 *   ./bin/icp_serial [n_points] [max_iters] [perturb] [kdtree|brute] [seed]
 *
 *   n_points   target cloud size            (default 50000)
 *   max_iters  ICP iteration cap            (default 50)
 *   perturb    ground-truth transform scale (default 1.0)
 *   backend    "kdtree" or "brute"          (default kdtree)
 *   seed       RNG seed                     (default 12345)
 *
 * Set ICP_DUMP=1 to also write target.pcd / source_initial.pcd /
 * source_aligned.pcd for tools/render_pcd.py.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L   /* clock_gettime; nvcc already sets 200809L */
#endif

#include "pointcloud.h"
#include "linalg.h"
#include "icp.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined (__AVX__) || defined (__AVX2__)
#include <immintrin.h>
#endif

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	int      n         = argc > 1 ? atoi(argv[1]) : 100000;
	int      max_iters = argc > 2 ? atoi(argv[2]) : 50;
	double   perturb   = argc > 3 ? atof(argv[3]) : 1.0;
	int      use_kd    = 1;
	if (argc > 4) {
		if      (strcmp(argv[4], "brute")  == 0) use_kd = 0;
		else if (strcmp(argv[4], "kdtree") == 0) use_kd = 1;
		else { fprintf(stderr, "backend must be 'kdtree' or 'brute'\n"); return 1; }
	}
	uint64_t seed = argc > 5 ? strtoull(argv[5], NULL, 10) : 12345ULL;

	if (n < 16) { fprintf(stderr, "n_points too small\n"); return 1; }

	rng_t rng;
	rng_seed(&rng, seed);

	/* --- target scene cloud --------------------------------------------- */
	double t0 = now_sec();
	PointCloud tgt;
	pc_generate_scene(&tgt, n, &rng);
	double t_gen = now_sec() - t0;

	/* --- ground-truth transform (tgt -> src) ---------------------------- */
	double Rgt[9];
	mat3_from_euler(0.08 * perturb, 0.12 * perturb, 0.15 * perturb, Rgt);
	double Tgt[3] = { 0.5 * perturb, -0.3 * perturb, 0.2 * perturb };

	/* --- source: transformed subset of target + noise + outliers -------- */
	PointCloud src;
	pc_make_source(&tgt, &src, Rgt, Tgt,
	               0.01,   /* noise std (m)      */
	               0.85,   /* keep fraction      */
	               0.05,   /* outlier fraction   */
	               &rng);

	/* --- run ICP -------------------------------------------------------- */
	ICPParams prm = {
		.max_iters     = max_iters,
		.tol           = 1e-6,
		.max_corr_dist = 1.0,
		.use_kdtree    = use_kd,
	};
	ICPResult res;
	t0 = now_sec();
	icp_run(&src, &tgt, &prm, &res);
	double t_icp = now_sec() - t0;

	/*
	 * ICP recovers the transform aligning src -> tgt. The ground truth maps
	 * tgt -> src, so the expected answer is its inverse:
	 *   R_expected = Rgt^T,  T_expected = -Rgt^T * Tgt.
	 */
	double Rexp[9];
	mat3_transpose(Rgt, Rexp);
	double Texp[3], tmp[3];
	mat3_mul_vec(Rexp, Tgt, tmp);
	Texp[0] = -tmp[0]; Texp[1] = -tmp[1]; Texp[2] = -tmp[2];

	double rot_err_deg = rot_geodesic(res.R, Rexp) * 180.0 / M_PI;
	double dT[3] = { res.T[0]-Texp[0], res.T[1]-Texp[1], res.T[2]-Texp[2] };
	double trans_err = sqrt(dT[0]*dT[0] + dT[1]*dT[1] + dT[2]*dT[2]);

	/* --- report --------------------------------------------------------- */
	printf("==== Serial ICP (point-to-point, Kabsch/SVD) ====\n");
	printf("backend         : %s\n", use_kd ? "KD-tree" : "brute force");
	printf("target points   : %d\n", tgt.n);
	printf("source points   : %d  (keep 85%%, +5%% outliers, noise 0.01 m)\n", src.n);
	printf("seed            : %llu\n", (unsigned long long)seed);
	printf("\n");
	printf("ground-truth T  : [% .4f % .4f % .4f]\n", Texp[0], Texp[1], Texp[2]);
	printf("recovered    T  : [% .4f % .4f % .4f]\n", res.T[0], res.T[1], res.T[2]);
	printf("\n");
	printf("iterations      : %d\n", res.iters);
	printf("final RMSE      : %.6f m  over %ld pairs\n", res.final_rmse, res.final_pairs);
	printf("rotation error  : %.4f deg\n", rot_err_deg);
	printf("translation err : %.4f m\n", trans_err);
	printf("\n");
	printf("---- timing (s) ----\n");
	printf("scene gen       : %8.4f\n", t_gen);
	printf("icp total       : %8.4f\n", t_icp);
	printf("  setup build   : %8.4f  (%.1f%%)\n", res.t_setup,
	       100.0 * res.t_setup / (t_icp > 0 ? t_icp : 1));
	if (res.t_ctx > 0)
		printf("  CUDA ctx init : %8.4f  (%.1f%%)\n", res.t_ctx,
		       100.0 * res.t_ctx / (t_icp > 0 ? t_icp : 1));
	if (res.t_upload > 0)
		printf("  H2D upload    : %8.4f  (%.1f%%)\n", res.t_upload,
		       100.0 * res.t_upload / (t_icp > 0 ? t_icp : 1));
	printf("  NN search     : %8.4f  (%.1f%%)\n", res.t_nn,
	       100.0 * res.t_nn / (t_icp > 0 ? t_icp : 1));
	printf("  reduce+solve  : %8.4f  (%.1f%%)\n", res.t_solve,
	       100.0 * res.t_solve / (t_icp > 0 ? t_icp : 1));
	printf("  transform     : %8.4f  (%.1f%%)\n", res.t_transform,
	       100.0 * res.t_transform / (t_icp > 0 ? t_icp : 1));

	int pass = (trans_err < 0.05 && rot_err_deg < 1.0);
	printf("\nresult          : %s\n", pass ? "PASS (converged to ground truth)"
	                                         : "CHECK (did not reach tight tolerance)");

	/* --- optional PCD dump ---------------------------------------------- */
	if (getenv("ICP_DUMP")) {
		pc_save_pcd(&tgt, "target.pcd", 1);
		pc_save_pcd(&src, "source_initial.pcd", 1);
		PointCloud aligned;
		pc_copy(&src, &aligned);
		pc_apply_transform(&aligned, res.R, res.T);
		pc_save_pcd(&aligned, "source_aligned.pcd", 1);
		pc_free(&aligned);
		printf("\nwrote target.pcd, source_initial.pcd, source_aligned.pcd\n");
	}

	pc_free(&tgt);
	pc_free(&src);
	return pass ? 0 : 2;
}
