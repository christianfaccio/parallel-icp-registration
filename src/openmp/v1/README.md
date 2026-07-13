Builds on v0. Once the NN search got fast, the previously negligible centroid +
cross-covariance reduction (stage 3 of the ICP loop) became a measurable chunk
of total time. This version parallelizes it with two more
`#pragma omp parallel for reduction(...)` regions (using OpenMP array-section
reductions for `cs`/`ct`/`H`), on top of the same query-level NN parallelism
and leaf-SIMD kd-tree from v0.
