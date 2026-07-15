#!/bin/bash
#SBATCH -J icp_omp_weak
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # must be >= max(OMP_THREADS_LIST); Ice Lake = 32 cores/socket
#SBATCH -t 00:40:00
#SBATCH -o weak_omp_%j.out -e weak_omp_%j.err
#
# Weak-scaling study for the OpenMP ICP backend: problem size grows with the
# thread count (n = WEAK_BASE * threads), so work-per-thread is held constant.
# Sweeps v0 (serial KD-tree build) and v2 (parallel task-based build) so their
# weak-scaling behaviour can be compared directly -- v0 is expected to droop as
# its serial build grows ~n log n, v2 to hold up. Plot both CSVs together with
# tools/plot_weak_scaling.py.
#
# One MPI-less, multi-threaded process; the sweep script picks OMP_NUM_THREADS
# itself, so do NOT export it here. Hand the allocated cores to the runtime.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"   # keep srun affinity sane if used

# Per-thread cloud size held constant across the sweep (total = WEAK_BASE*threads).
export WEAK_BASE="${WEAK_BASE:-50000}"

# Which versioned OpenMP backends to build and weak-scale.
VERSIONS="${VERSIONS:-v0 v2}"

make $(for v in $VERSIONS; do echo omp_$v; done)

for v in $VERSIONS; do
	bash scripts/run_omp_weak.sh "out/openmp/weak_$v.csv" "$v" 50 12345
done
