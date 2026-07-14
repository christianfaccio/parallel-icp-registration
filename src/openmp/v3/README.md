Builds on v1 (parallel NN search + parallel reduction). This version also
parallelizes the KD-tree *build* (`kd_build`/`build_rec`) with OpenMP tasks:
after `qselect` partitions a node's point range into a left and a right half,
those two halves are disjoint and independent, so they're spawned as sibling
`#pragma omp task`s instead of recursed into serially. Subtrees smaller than
`TASK_MIN` points fall back to serial recursion, since task creation overhead
dominates near the leaves.

The one thing that needs explicit protection is the shared node-pool counter
(`next`): every task that allocates a node index does so with
`#pragma omp atomic capture`. Point permutation writes need no such
protection -- sibling tasks always operate on disjoint slices of `perm`.
