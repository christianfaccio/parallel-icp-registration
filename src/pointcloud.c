/*
 * pointcloud.c -- SoA point cloud container, synthetic scene generation,
 * the rigid-transform kernel, and PCD output.
 */

#include "pointcloud.h"
#include "linalg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void pc_init(PointCloud *c)
{
	c->x = c->y = c->z = NULL;
	c->n = c->cap = 0;
}

void pc_free(PointCloud *c)
{
	free(c->x); free(c->y); free(c->z);
	pc_init(c);
}

void pc_reserve(PointCloud *c, int n)
{
	if (n <= c->cap) return;
	int cap = c->cap ? c->cap : 1024;
	while (cap < n) cap *= 2;
	c->x = realloc(c->x, (size_t)cap * sizeof *c->x);
	c->y = realloc(c->y, (size_t)cap * sizeof *c->y);
	c->z = realloc(c->z, (size_t)cap * sizeof *c->z);
	if (!c->x || !c->y || !c->z) { perror("pc_reserve"); exit(1); }
	c->cap = cap;
}

void pc_push(PointCloud *c, float x, float y, float z)
{
	if (c->n == c->cap) pc_reserve(c, c->cap ? c->cap * 2 : 1024);
	c->x[c->n] = x; c->y[c->n] = y; c->z[c->n] = z;
	c->n++;
}

void pc_copy(const PointCloud *src, PointCloud *dst)
{
	pc_init(dst);
	pc_reserve(dst, src->n);
	memcpy(dst->x, src->x, (size_t)src->n * sizeof *src->x);
	memcpy(dst->y, src->y, (size_t)src->n * sizeof *src->y);
	memcpy(dst->z, src->z, (size_t)src->n * sizeof *src->z);
	dst->n = src->n;
}

void pc_bounds(const PointCloud *c, double bmin[3], double bmax[3])
{
	if (c->n == 0) {
		bmin[0] = bmin[1] = bmin[2] = 0.0;
		bmax[0] = bmax[1] = bmax[2] = 0.0;
		return;
	}
	bmin[0] = bmax[0] = c->x[0];
	bmin[1] = bmax[1] = c->y[0];
	bmin[2] = bmax[2] = c->z[0];
	for (int i = 1; i < c->n; i++) {
		if (c->x[i] < bmin[0]) bmin[0] = c->x[i];
		if (c->x[i] > bmax[0]) bmax[0] = c->x[i];
		if (c->y[i] < bmin[1]) bmin[1] = c->y[i];
		if (c->y[i] > bmax[1]) bmax[1] = c->y[i];
		if (c->z[i] < bmin[2]) bmin[2] = c->z[i];
		if (c->z[i] > bmax[2]) bmax[2] = c->z[i];
	}
}

/* random point on a sphere of radius r about (cx,cy,cz) */
static void sample_sphere(rng_t *rng, double cx, double cy, double cz, double r,
                          double *x, double *y, double *z)
{
	double dx = rng_normal(rng, 0, 1);
	double dy = rng_normal(rng, 0, 1);
	double dz = rng_normal(rng, 0, 1);
	double nn = sqrt(dx*dx + dy*dy + dz*dz) + 1e-12;
	*x = cx + r * dx / nn;
	*y = cy + r * dy / nn;
	*z = cz + r * dz / nn;
}

void pc_generate_scene(PointCloud *c, int n, rng_t *rng)
{
	pc_init(c);
	pc_reserve(c, n);

	const double L = 10.0;   /* room footprint (x,y in [0,L]) */
	const double H = 4.0;    /* room height   (z in [0,H])    */

	for (int i = 0; i < n; i++) {
		double u = rng_uniform(rng);
		double x, y, z;
		if (u < 0.25) {                 /* floor  z = 0 */
			x = rng_uniform(rng) * L; y = rng_uniform(rng) * L; z = 0.0;
		} else if (u < 0.45) {          /* wall x = 0 */
			x = 0.0; y = rng_uniform(rng) * L; z = rng_uniform(rng) * H;
		} else if (u < 0.65) {          /* wall y = 0 */
			y = 0.0; x = rng_uniform(rng) * L; z = rng_uniform(rng) * H;
		} else if (u < 0.80) {          /* ceiling z = H */
			x = rng_uniform(rng) * L; y = rng_uniform(rng) * L; z = H;
		} else if (u < 0.90) {          /* sphere 1 */
			sample_sphere(rng, 3.0, 3.0, 1.5, 1.0, &x, &y, &z);
		} else {                        /* sphere 2 */
			sample_sphere(rng, 7.0, 6.0, 1.0, 0.8, &x, &y, &z);
		}
		pc_push(c, (float)x, (float)y, (float)z);
	}
}

void pc_make_source(const PointCloud *tgt, PointCloud *src,
                    const double R[9], const double T[3],
                    double noise_std, double keep_frac, double outlier_frac,
                    rng_t *rng)
{
	pc_init(src);
	pc_reserve(src, tgt->n);

	for (int i = 0; i < tgt->n; i++) {
		if (rng_uniform(rng) > keep_frac) continue;   /* partial overlap */
		double p[3] = { tgt->x[i], tgt->y[i], tgt->z[i] };
		double q[3];
		mat3_mul_vec(R, p, q);
		double x = q[0] + T[0] + rng_normal(rng, 0, noise_std);
		double y = q[1] + T[1] + rng_normal(rng, 0, noise_std);
		double z = q[2] + T[2] + rng_normal(rng, 0, noise_std);
		pc_push(src, (float)x, (float)y, (float)z);
	}

	/* sprinkle uniform outliers inside the (transformed) bounding box */
	double bmin[3], bmax[3];
	pc_bounds(src, bmin, bmax);
	int n_out = (int)(outlier_frac * src->n);
	for (int i = 0; i < n_out; i++) {
		double x = bmin[0] + rng_uniform(rng) * (bmax[0] - bmin[0]);
		double y = bmin[1] + rng_uniform(rng) * (bmax[1] - bmin[1]);
		double z = bmin[2] + rng_uniform(rng) * (bmax[2] - bmin[2]);
		pc_push(src, (float)x, (float)y, (float)z);
	}
}

void pc_apply_transform(PointCloud *c, const double R[9], const double T[3])
{
	/* Hot, embarrassingly-parallel kernel: each point is independent.
	 * This is the natural first SIMD / OpenMP target. */
	for (int i = 0; i < c->n; i++) {
		double x = c->x[i], y = c->y[i], z = c->z[i];
		c->x[i] = (float)(R[0]*x + R[1]*y + R[2]*z + T[0]);
		c->y[i] = (float)(R[3]*x + R[4]*y + R[5]*z + T[1]);
		c->z[i] = (float)(R[6]*x + R[7]*y + R[8]*z + T[2]);
	}
}

void pc_save_pcd(const PointCloud *c, const char *path, int stride)
{
	if (stride < 1) stride = 1;
	FILE *f = fopen(path, "w");
	if (!f) { perror(path); return; }

	int count = 0;
	for (int i = 0; i < c->n; i += stride) count++;

	fprintf(f, "# .PCD v.7 - Point Cloud Data file format\n");
	fprintf(f, "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n");
	fprintf(f, "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n", count);
	fprintf(f, "POINTS %d\nDATA ascii\n", count);
	for (int i = 0; i < c->n; i += stride)
		fprintf(f, "%f %f %f\n", c->x[i], c->y[i], c->z[i]);

	fclose(f);
}
