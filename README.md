# 3D Point Cloud Registration via ICP — Serial Baseline

**Author:** Christian Faccio · Advanced HPC, University of Trieste

Frame-to-frame LiDAR registration: find the rigid transform `(R, T)` that aligns
a **source** point cloud onto a **target** point cloud. This is how a moving
vehicle tracks its own motion between scans. The point of the project is that
ICP's pipeline has three architecturally distinct bottlenecks, each mapping to a
different course technology and producing interesting scaling behaviour.

This folder is the **serial correctness baseline** — the clean starting point
for the parallelization work.

## The algorithm (point-to-point ICP, Kabsch/SVD)

Iterate until the RMSE stops improving:

```
[1. Nearest Neighbour]  ->  [2. Reject outliers]  ->  [3. Reduce: centroids + H]  ->  [4. SVD solve + apply]
   (dominant cost)            (data-dependent)           (global sums)                  (3x3 Kabsch)
```

1. **NN search** — for each source point, find the closest target point
   (KD-tree, or brute force as oracle).
2. **Rejection** — drop correspondences farther than `max_corr_dist`.
3. **Reduction** — centroids of both matched sets and the `3x3` cross-covariance
   `H = Σ (sᵢ−c_s)(tᵢ−c_t)ᵀ`.
4. **Solve** — SVD of `H` (hand-rolled one-sided Jacobi), `R = V·Uᵀ` with the
   reflection fix, `T = c_t − R·c_s`; apply to the source and accumulate the
   total transform.

## Layout

```
include/   rng.h  linalg.h  pointcloud.h  kdtree.h  icp.h
src/       linalg.c  pointcloud.c  kdtree.c  icp.c  main.c
tools/     render_pcd.py     # stdlib-only PNG previewer
Makefile   README.md
```

- `PointCloud` is **Struct-of-Arrays** (`x[]`, `y[]`, `z[]`) so the kernels stay
  SIMD/GPU-friendly.
- `rng_t` is a **reentrant** splitmix64 RNG — deterministic now, thread-safe later.
- All geometry is self-contained (the lecture notes don't cover KD-trees/SVD).

## Build & run

Inside the course Docker container (`~/Linux/`):

```bash
make                       # builds bin/icp_serial  (-O3 -march=native, warning-free)
make run                   # default run
./bin/icp_serial 100000 50 1.0 kdtree 12345
./bin/icp_serial 20000 50 1.0 brute       # brute-force oracle
make asan && ./bin/icp_serial 20000 50    # leak / UB check
```

Arguments: `n_points  max_iters  perturb  [kdtree|brute]  seed`.

The run reports recovered vs. ground-truth `(R, T)` (rotation error in degrees,
translation in metres), iteration count, final RMSE, and a **per-stage timing
breakdown** — confirm NN search dominates (that's what motivates the work).

Visual check:

```bash
ICP_DUMP=1 ./bin/icp_serial 30000
python3 tools/render_pcd.py preview.png   # gray=target, red=initial, green=aligned
```

## Data

Synthetic, with a **known ground-truth transform** so accuracy is exactly
measurable and `n_points` can be dialled from 1k to 1M+ for scaling graphs. The
target is a room corner (floor, ceiling, two walls) + two spheres (full 6-DOF
observability); the source is a transformed, noisy, partially-overlapping subset
with 5% injected outliers.

## Status & parallelization roadmap

- [x] **Serial baseline** — point-to-point ICP, KD-tree + brute-force NN,
      hand-rolled 3×3 SVD, ground-truth validation, per-stage timing.
- [ ] **SIMD (Ch1)** — vectorize the per-point transform `R·p + T` and the
      distance evaluations (SoA already in place).
- [ ] **OpenMP (Ch2/3)** — parallelize the NN loop with `schedule(dynamic)`
      (traversal cost is uneven), and the centroid/`H` accumulation with
      `reduction` / `task_reduction`. RNG is already per-thread-safe.
- [ ] **MPI (Ch4)** — partition the source across ranks; `MPI_Allreduce` the
      partial centroids and `H`; mitigate the per-iteration reduction latency
      with asynchronous / one-sided communication.
- [ ] **CUDA / OpenACC (Ch6/7)** — offload NN search (attack warp divergence:
      flatten the KD-tree to an array or switch to a voxel grid) and the
      reduction (warp-shuffle `__shfl_down_sync` block reductions).

## Notes for the parallel phases

- **NN search is the bottleneck** and the most interesting target: the
  pointer-/index-chasing KD-tree traversal has poor cache locality (CPU) and
  causes warp divergence (GPU). Flattening the tree or switching to a voxel grid
  is the canonical fix.
- **Outlier rejection** creates data-dependent, divergent control flow — relevant
  to both SIMD masking and GPU warp efficiency.
- **The reduction** (`centroids`, `H`) is the global-synchronization point that
  will dominate MPI scaling once NN is fast; the SVD itself is a fixed `3×3` and
  stays on one rank/thread.
# parallel-icp-registration
