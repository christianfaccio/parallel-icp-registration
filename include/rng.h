#ifndef ICP_RNG_H_
#define ICP_RNG_H_

/*
 * Reentrant random number generator (splitmix64 core).
 *
 * Header-only and self-contained: each rng_t carries its own state, so there
 * is no global rand()-style hidden state. This matters for two reasons later:
 *   1. data generation is fully deterministic given a seed (reproducible runs);
 *   2. the OpenMP / CUDA backends can give each thread its own rng_t with no
 *      locking or data races.
 */

#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { uint64_t s; } rng_t;

static inline void rng_seed(rng_t *r, uint64_t seed)
{
	/* avoid the all-zero state, which splitmix64 handles but is a poor seed */
	r->s = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

/* splitmix64: fast, good statistical quality, trivially parallel */
static inline uint64_t rng_next_u64(rng_t *r)
{
	uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* uniform double in [0, 1) using the top 53 bits */
static inline double rng_uniform(rng_t *r)
{
	return (rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}

/* Gaussian sample ~ N(mu, sigma) via Box-Muller */
static inline double rng_normal(rng_t *r, double mu, double sigma)
{
	double u1 = rng_uniform(r);
	double u2 = rng_uniform(r);
	if (u1 < 1e-300) u1 = 1e-300;        /* guard log(0) */
	double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	return mu + sigma * z;
}

#endif /* ICP_RNG_H_ */
