v2 - built on top of v1 but with optimizations on memory and cache locality.

After profiling the cuda code, I saw that only 9/32 warps are active at a time (most are masked due to divergence). Cache locality is the main problem, since warps are most of the time idle waiting for global memory loads.


