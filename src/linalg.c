/*
 * linalg.c -- 3x3 matrix ops, a one-sided Jacobi SVD, and the Kabsch solve.
 * See linalg.h for the storage convention (row-major double[9]).
 */

#include "linalg.h"
#include <math.h>

void mat3_identity(double M[9])
{
	M[0]=1; M[1]=0; M[2]=0;
	M[3]=0; M[4]=1; M[5]=0;
	M[6]=0; M[7]=0; M[8]=1;
}

void mat3_mul(const double A[9], const double B[9], double C[9])
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			C[i*3 + j] = A[i*3+0]*B[0*3+j]
			           + A[i*3+1]*B[1*3+j]
			           + A[i*3+2]*B[2*3+j];
}

void mat3_transpose(const double A[9], double At[9])
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			At[i*3 + j] = A[j*3 + i];
}

void mat3_mul_vec(const double A[9], const double v[3], double out[3])
{
	out[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];
	out[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];
	out[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];
}

double mat3_det(const double A[9])
{
	return A[0]*(A[4]*A[8] - A[5]*A[7])
	     - A[1]*(A[3]*A[8] - A[5]*A[6])
	     + A[2]*(A[3]*A[7] - A[4]*A[6]);
}

void mat3_from_euler(double rx, double ry, double rz, double R[9])
{
	double cx = cos(rx), sx = sin(rx);
	double cy = cos(ry), sy = sin(ry);
	double cz = cos(rz), sz = sin(rz);

	double Rx[9] = { 1,0,0,  0,cx,-sx,  0,sx,cx };
	double Ry[9] = { cy,0,sy,  0,1,0,  -sy,0,cy };
	double Rz[9] = { cz,-sz,0,  sz,cz,0,  0,0,1 };

	double Rzy[9];
	mat3_mul(Rz, Ry, Rzy);
	mat3_mul(Rzy, Rx, R);   /* R = Rz * Ry * Rx */
}

double rot_geodesic(const double A[9], const double B[9])
{
	double At[9], M[9];
	mat3_transpose(A, At);
	mat3_mul(At, B, M);
	double tr = M[0] + M[4] + M[8];
	double c = 0.5 * (tr - 1.0);
	if (c >  1.0) c =  1.0;
	if (c < -1.0) c = -1.0;
	return acos(c);
}

/*
 * One-sided Jacobi SVD of a 3x3.
 *
 * Idea: post-multiply a working copy W = H by Givens rotations until its
 * columns become mutually orthogonal (W <- W*G, V <- V*G). Once orthogonal,
 *   sigma_j = ||W_col_j||,   U_col_j = W_col_j / sigma_j,   H = U S V^T.
 * For a 3x3 a handful of sweeps over the three column pairs converges to
 * machine precision.
 */
void svd3(const double H[9], double U[9], double S[3], double V[9])
{
	double W[9];
	for (int i = 0; i < 9; i++) W[i] = H[i];
	mat3_identity(V);

	const int pairs[3][2] = { {0,1}, {0,2}, {1,2} };

	for (int sweep = 0; sweep < 30; sweep++) {
		double off = 0.0;
		for (int p = 0; p < 3; p++) {
			int i = pairs[p][0], j = pairs[p][1];
			double alpha = 0, beta = 0, gamma = 0;
			for (int k = 0; k < 3; k++) {
				double wi = W[k*3+i], wj = W[k*3+j];
				alpha += wi*wi;
				beta  += wj*wj;
				gamma += wi*wj;
			}
			off += fabs(gamma);
			if (fabs(gamma) < 1e-300) continue;

			/* Jacobi rotation diagonalizing [[alpha,gamma],[gamma,beta]] */
			double zeta = (beta - alpha) / (2.0 * gamma);
			double t = (zeta >= 0 ? 1.0 : -1.0)
			         / (fabs(zeta) + sqrt(zeta*zeta + 1.0));
			double c = 1.0 / sqrt(t*t + 1.0);
			double s = c * t;

			for (int k = 0; k < 3; k++) {
				double wi = W[k*3+i], wj = W[k*3+j];
				W[k*3+i] = c*wi - s*wj;
				W[k*3+j] = s*wi + c*wj;
				double vi = V[k*3+i], vj = V[k*3+j];
				V[k*3+i] = c*vi - s*vj;
				V[k*3+j] = s*vi + c*vj;
			}
		}
		if (off < 1e-15) break;
	}

	/* singular values and U columns from the orthogonalized W */
	double sigma[3];
	double Ucol[3][3];   /* Ucol[j][k] = U entry (row k, col j) */
	for (int j = 0; j < 3; j++) {
		double n2 = 0;
		for (int k = 0; k < 3; k++) { double w = W[k*3+j]; n2 += w*w; }
		double n = sqrt(n2);
		sigma[j] = n;
		if (n > 1e-12)
			for (int k = 0; k < 3; k++) Ucol[j][k] = W[k*3+j] / n;
		else
			for (int k = 0; k < 3; k++) Ucol[j][k] = 0.0;
	}
	/* repair any degenerate U column so U stays orthonormal */
	for (int j = 0; j < 3; j++) {
		if (sigma[j] > 1e-12) continue;
		int a = (j+1)%3, b = (j+2)%3;
		Ucol[j][0] = Ucol[a][1]*Ucol[b][2] - Ucol[a][2]*Ucol[b][1];
		Ucol[j][1] = Ucol[a][2]*Ucol[b][0] - Ucol[a][0]*Ucol[b][2];
		Ucol[j][2] = Ucol[a][0]*Ucol[b][1] - Ucol[a][1]*Ucol[b][0];
	}

	/* order columns by descending singular value */
	int ord[3] = {0, 1, 2};
	for (int a = 0; a < 3; a++)
		for (int b = a+1; b < 3; b++)
			if (sigma[ord[b]] > sigma[ord[a]]) {
				int tmp = ord[a]; ord[a] = ord[b]; ord[b] = tmp;
			}

	double Vtmp[9];
	for (int k = 0; k < 9; k++) Vtmp[k] = V[k];
	for (int c2 = 0; c2 < 3; c2++) {
		int o = ord[c2];
		S[c2] = sigma[o];
		for (int k = 0; k < 3; k++) {
			U[k*3 + c2] = Ucol[o][k];
			V[k*3 + c2] = Vtmp[k*3 + o];
		}
	}
}

void kabsch_from_H(const double H[9], const double cen_src[3],
                   const double cen_tgt[3], double R[9], double T[3])
{
	double U[9], S[3], V[9], Ut[9];
	svd3(H, U, S, V);
	mat3_transpose(U, Ut);
	mat3_mul(V, Ut, R);              /* R = V * U^T */

	/* reflection fix: a negative determinant means SVD picked an improper
	 * rotation; flip the sign of V's last column and recompute. */
	if (mat3_det(R) < 0.0) {
		V[0*3+2] = -V[0*3+2];
		V[1*3+2] = -V[1*3+2];
		V[2*3+2] = -V[2*3+2];
		mat3_mul(V, Ut, R);
	}

	double Rc[3];
	mat3_mul_vec(R, cen_src, Rc);    /* T = cen_tgt - R*cen_src */
	T[0] = cen_tgt[0] - Rc[0];
	T[1] = cen_tgt[1] - Rc[1];
	T[2] = cen_tgt[2] - Rc[2];
}
