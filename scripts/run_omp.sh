#!/usr/bin/env bash
#
# Strong-scaling sweep for the OpenMP ICP backend (bin/openmp/icp_openmp_vN).
#
# For each thread count in THREADS and each problem size in NS it runs the
# solver and records timing, emitting ONE CSV with a leading `threads` column:
#
#   threads,n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query
#
# With a fixed n, reading down the thread counts gives strong scaling; compute
# speedup = t_icp(1) / t_icp(p) and efficiency = speedup / p in the plot step.
#
# IMPORTANT: unlike run_serial.sh this does NOT taskset-pin to one core -- that
# would cram every thread onto a single CPU. Thread placement is controlled with
# OMP_PLACES=cores + OMP_PROC_BIND=close so runs are reproducible. Make sure the
# SLURM allocation gives you at least max(THREADS) cores (--cpus-per-task).
#
# Usage:
#   make omp_v0                  # or omp_v1, ...
#   scripts/run_omp.sh [output.csv] [version] [max_iters] [seed]
#
# Defaults: out=out/openmp/omp_<version>.csv, version=v0, max_iters=50, seed=12345.
# Override the sweeps from the environment, e.g.:
#   OMP_THREADS_LIST="1 2 4 8 16 32" NS_LIST="46811 150000" scripts/run_omp.sh

set -euo pipefail
export LC_ALL=C   # force '.' decimal separator so awk printf doesn't emit commas

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="${2:-v0}"
BIN="$HERE/bin/openmp/icp_openmp_$VERSION"

OUT="${1:-out/openmp/omp_$VERSION.csv}"
MAX_ITERS="${3:-50}"
SEED="${4:-12345}"

if [[ ! -x "$BIN" ]]; then
	echo "error: $BIN not found. Build it first with: make omp_$VERSION" >&2
	exit 1
fi

# Thread counts and problem sizes (override via env). Threading overhead
# dominates on tiny clouds, so the size list starts where it begins to pay off.
THREADS="${OMP_THREADS_LIST:-1 2 4 8 16 32}"
read -r -a NS <<< "${NS_LIST:-500000}"

# Reproducible thread placement: one thread per physical core, bound.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p "$(dirname "$OUT")"
echo "threads,n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query" > "$OUT"
echo "# OMP sweep: threads={$THREADS}, ${#NS[@]} sizes -> $OUT" >&2

for th in $THREADS; do
	export OMP_NUM_THREADS="$th"
	for n in "${NS[@]}"; do
		printf 'threads=%-3s n=%-9s ... ' "$th" "$n" >&2
		out="$("$BIN" "$n" "$MAX_ITERS" 1.0 kdtree "$SEED")"

		tgt=$(  awk -F: '/^target points/ {gsub(/ /,"",$2); print $2}'        <<<"$out")
		src=$(  awk -F: '/^source points/ {print $2}'                         <<<"$out" | awk '{print $1}')
		iters=$(awk -F: '/^iterations/    {gsub(/ /,"",$2); print $2}'        <<<"$out")
		tnn=$(  awk        '/NN search/    {print $4}'                        <<<"$out")
		pct=$(  awk        '/NN search/    {gsub(/[()%]/,"",$5); print $5}'   <<<"$out")
		ticp=$( awk        '/^icp total/   {print $4}'                        <<<"$out")

		nspq=$(awk -v t="$tnn" -v it="$iters" -v s="$src" \
			'BEGIN { d = it*s; printf (d>0 ? "%.3f" : "nan"), (d>0 ? t/d*1e9 : 0) }')

		echo "$th,$n,$tgt,$src,$iters,$tnn,$ticp,$pct,$nspq" >> "$OUT"
		printf 'src=%-8s iters=%-3s t_icp=%-9s ns/query=%s\n' "$src" "$iters" "$ticp" "$nspq" >&2
	done
done

echo "done -> $OUT" >&2
