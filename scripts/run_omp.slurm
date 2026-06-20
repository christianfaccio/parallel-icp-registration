#!/bin/bash
#SBATCH -J icp_omp
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # must be >= max(OMP_THREADS_LIST); Ice Lake = 32 cores/socket
#SBATCH -t 00:30:00
#SBATCH -o omp_%j.out -e omp_%j.err

# One MPI-less, multi-threaded process. The script sweeps OMP_NUM_THREADS itself,
# so do NOT export it here. Hand the allocated cores to the OpenMP runtime.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"   # keep srun affinity sane if used

bash scripts/run_omp.sh v0 "out/openmp/omp.csv" 50 12345
