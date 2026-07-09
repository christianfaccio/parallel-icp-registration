Here openmp is used to spawn multiple threads that run in parallel through queries,
so that each one traverses the tree independently and uses the SIMD capabilities
developed in the vectorization part.
