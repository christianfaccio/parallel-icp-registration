#ifndef ICP_LINALG_H_
#define ICP_LINALG_H_

/*
 * Small dense linear algebra for ICP: 3-vectors, 3x3 matrices, a hand-rolled
 * 3x3 SVD, and the Kabsch rigid-transform solve.
 */

typedef struct { double x, y, z; } Vec3;

static inline Vec3   v3(double x, double y, double z) { return (Vec3){x, y, z}; }
static inline Vec3   v3_add(Vec3 a, Vec3 b)  { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3   v3_sub(Vec3 a, Vec3 b)  { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3   v3_scale(Vec3 a, double s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline double v3_dot(Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }

/* ---- 3x3 matrix ops (row-major double[9]) ---------------------------------- */

void   mat3_identity(double M[9]);
void   mat3_mul(const double A[9], const double B[9], double C[9]);  /* C = A * B */
void   mat3_transpose(const double A[9], double At[9]);
void   mat3_mul_vec(const double A[9], const double v[3], double out[3]); /* out = A*v */
double mat3_det(const double A[9]);

/* Rotation matrix from intrinsic XYZ Euler angles (R = Rz * Ry * Rx). */
void   mat3_from_euler(double rx, double ry, double rz, double R[9]);

/* Geodesic angle (radians) between two rotation matrices: acos((tr(A^T B)-1)/2). */
double rot_geodesic(const double A[9], const double B[9]);

/* ---- Decompositions -------------------------------------------------------- */

/*
 * One-sided Jacobi SVD of a 3x3 matrix: H = U * diag(S) * V^T, with the
 * singular values S sorted in descending order and U, V orthonormal.
 */
void svd3(const double H[9], double U[9], double S[3], double V[9]);

/*
 * Kabsch / Umeyama rigid solve. Given the cross-covariance
 *   H = sum_i (src_i - cen_src) (tgt_i - cen_tgt)^T
 * and the two centroids, returns the rotation R and translation T that
 * minimize sum_i | R*src_i + T - tgt_i |^2. Includes the reflection fix
 * (det(R) < 0) so the result is always a proper rotation.
 */
void kabsch_from_H(const double H[9], const double cen_src[3],
                   const double cen_tgt[3], double R[9], double T[3]);

#endif /* ICP_LINALG_H_ */
