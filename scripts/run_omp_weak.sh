#!/usr/bin/env bash
#
# Weak-scaling sweep for the OpenMP ICP backend (bin/openmp/icp_openmp_vN).
#
# Weak scaling holds the work-per-thread constant: the problem size grows in
# lock-step with the thread count, n(p) = BASE * p. So each thread always owns
# ~BASE source points and, ideally, wall time stays flat as p (and n) grow.
# Contrast run_omp.sh, which fixes n and grows p (strong scaling).
#
# One CSV is emitted with a leading `threads` column (same schema as the
# strong-scaling sweep, so tools/ loaders are shared):
#
#   threads,n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query
#
# Reading down the rows: n climbs with threads. Ideal weak scaling => t_icp_s
# and ns_per_query stay flat; weak efficiency t(p0)/t(p) stays at 1. The KD-tree
# breaks the ideal on two fronts, which is the point of measuring it: per-query
# traversal grows ~log(n), and any *serial* tree build (v0) grows ~n log n and
# eats a Gustafson-style non-scalable slice that widens with n. v2 parallelizes
# the build, so it should hold up better -- exactly the v0-vs-v2 contrast asked.
#
# IMPORTANT: like run_omp.sh this does NOT taskset-pin to one core -- that would
# cram every thread onto a single CPU. Placement is fixed with OMP_PLACES=cores
# + OMP_PROC_BIND=close. The SLURM allocation must give at least max(THREADS)
# cores (--cpus-per-task).
#
# Usage:
#   make omp_v0                  # or omp_v2, ...
#   scripts/run_omp_weak.sh [output.csv] [version] [max_iters] [seed]
#
# Defaults: out=out/openmp/weak_<version>.csv, version=v0, max_iters=50, seed=12345.
# Override the sweeps from the environment, e.g.:
#   OMP_THREADS_LIST="1 2 4 8 16 32" WEAK_BASE=50000 scripts/run_omp_weak.sh

set -euo pipefail
export LC_ALL=C   # force '.' decimal separator so awk printf doesn't emit commas

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="${2:-v0}"
BIN="$HERE/bin/openmp/icp_openmp_$VERSION"

OUT="${1:-out/openmp/weak_$VERSION.csv}"
MAX_ITERS="${3:-50}"
SEED="${4:-12345}"

if [[ ! -x "$BIN" ]]; then
	echo "error: $BIN not found. Build it first with: make omp_$VERSION" >&2
	exit 1
fi

# Thread counts, and the per-thread problem size that stays constant across the
# sweep (override via env). Total target points at p threads is WEAK_BASE * p.
THREADS="${OMP_THREADS_LIST:-1 2 4 8 16 32}"
BASE="${WEAK_BASE:-50000}"

# Reproducible thread placement: one thread per physical core, bound.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p "$(dirname "$OUT")"
echo "threads,n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query" > "$OUT"
echo "# OMP weak sweep ($VERSION): threads={$THREADS}, n=${BASE}*threads -> $OUT" >&2

for th in $THREADS; do
	export OMP_NUM_THREADS="$th"
	n=$(( BASE * th ))                    # grow the cloud with the thread count
	printf 'threads=%-3s n=%-9s ... ' "$th" "$n" >&2
	out="$("$BIN" "$n" "$MAX_ITERS" 1.0 kdtree "$SEED")"

	tgt=$(  awk -F: '/^target points/ {gsub(/ /,"",$2); print $2}'        <<<"$out")
	src=$(  awk -F: '/^source points/ {print $2}'                         <<<"$out" | awk '{print $1}')
	iters=$(awk -F: '/^iterations/    {gsub(/ /,"",$2); print $2}'        <<<"$out")
	tnn=$(  awk        '/NN search/    {print $4}'                        <<<"$out")
	pct=$(  awk        '/NN search/    {gsub(/[()%]/,"",$5); print $5}'   <<<"$out")
	ticp=$( awk        '/^icp total/   {print $4}'                        <<<"$out")

	# ns per query per iteration: the natural weak-scaling latency metric. With
	# work-per-thread fixed, perfect scaling keeps this flat as p (and n) grow.
	nspq=$(awk -v t="$tnn" -v it="$iters" -v s="$src" \
		'BEGIN { d = it*s; printf (d>0 ? "%.3f" : "nan"), (d>0 ? t/d*1e9 : 0) }')

	echo "$th,$n,$tgt,$src,$iters,$tnn,$ticp,$pct,$nspq" >> "$OUT"
	printf 'src=%-8s iters=%-3s t_icp=%-9s ns/query=%s\n' "$src" "$iters" "$ticp" "$nspq" >&2
done

echo "done -> $OUT" >&2
