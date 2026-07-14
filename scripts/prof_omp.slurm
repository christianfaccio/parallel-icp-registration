#!/bin/bash
#SBATCH -J icp_omp
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # must be >= THREADS; Ice Lake = 32 cores/socket
#SBATCH -t 00:20:00
#SBATCH -o omp_%j.out -e omp_%j.err
#
# perf profiling for the OpenMP ICP backend. Unlike prof.slurm (single-threaded
# vec binaries), this must NOT pin the whole process to one core -- that would
# serialize every OMP thread onto a single CPU and make the counters meaningless.
# Instead: request enough cores for THREADS, bind one thread per core, and let
# perf attach to the whole process (no -C) so it aggregates across all threads.
#
# Which versioned OpenMP backend to build and profile (v0, v1, ...), and how
# many threads to run with.
VERSION="${VERSION:-v1}"
THREADS="${THREADS:-32}"
N="${N:-500000}"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

BIN="bin/openmp/icp_openmp_$VERSION"

perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses "$BIN" "$N"

perf record -g --call-graph dwarf -F 999 "$BIN" "$N"

perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > "omp_${VERSION}_flame.svg"
