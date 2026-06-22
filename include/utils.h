#ifndef ICP_UTILS_H_
#define ICP_UTILS_H_

#include <stdlib.h>
#include <stdint.h>

typedef struct
{
	uint64_t code;
	int idx;
}MortonKey;

// spread the low 21 bits of x so each bit has two zero gaps after it
static inline uint64_t spread3(uint32_t v) {	// v uses only low 21 bits
	uint64_t x = v & 0x1fffff;             	// mask to 21 bits
      	x = (x | x << 32) & 0x1f00000000ffffULL;
        x = (x | x << 16) & 0x1f0000ff0000ffULL;
        x = (x | x << 8)  & 0x100f00f00f00f00fULL;
        x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
        x = (x | x << 2)  & 0x1249249249249249ULL;
        return x;
}

static inline uint32_t quantize(float x, float min, float max)
{
	float range = max - min;
	if (range <= 0.0f) return 0;			// degenerate axis
	float t = (x - min) / range;			// normalize to [0, 1]
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	return (uint32_t)(t * 2097151.0f);		// 2^21 - 1, stays in 21 bits
}

static inline int cmp_morton(const void *a, const void *b)
{
	uint64_t ca = ((const MortonKey*)a)->code;
	uint64_t cb = ((const MortonKey*)b)->code;
	return (ca > cb) - (ca < cb);
}

#endif /* ICP_UTILS_H_ */
