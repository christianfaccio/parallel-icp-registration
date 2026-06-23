#!/usr/bin/env bash
#
# Runs bin/cuda/icp_cuda_<ver> over a geometric sweep of n and emits a CSV whose
# key column is the AMORTIZED time per nearest-neighbour query:
#
#     ns_per_query = t_nn / (iters * source_points) * 1e9
#
# Same CSV schema as the serial/vectorized sweeps, so tools/plot_compare.py can
# overlay the GPU curve against the CPU backends directly.
#
# Unlike the CPU sweeps there is no taskset pinning: the host side is a thin
# driver and the work runs on the GPU. The sweep is extended to larger n because
# the GPU only pays off once there is enough parallel work to fill it.
#
# Usage:
#     make cuda_v0                 # or cuda_v1, ...
#     scripts/run_cuda.sh [output.csv] [version] [max_iters] [seed]
#
# Defaults: out=out/cuda/cuda_<version>.csv, version=v0, max_iters=50, seed=12345.
# On Leonardo, run inside an salloc'd / sbatch'd GPU node (see run_cuda.slurm),
# never the login node.

set -euo pipefail

# Force '.' as decimal separator so awk's printf doesn't emit locale commas
# (which would corrupt the CSV).
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="${2:-v0}"
BIN="$HERE/bin/cuda/icp_cuda_$VERSION"

OUT="${1:-out/cuda/cuda_$VERSION.csv}"
MAX_ITERS="${3:-50}"
SEED="${4:-12345}"

if [[ ! -x "$BIN" ]]; then
	echo "error: $BIN not found. Build it first with: make cuda_$VERSION" >&2
	exit 1
fi

mkdir -p "$(dirname "$OUT")"

# Geometric sweep. Same lower range as the CPU sweeps for comparability, plus a
# few larger sizes where the GPU has enough work to amortize launch + transfer.
NS=(
	500 1000 1755 3000 6000 12000 25000 46811 80000 100000 150000 250000 500000
)

echo "n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query" > "$OUT"
echo "# sweeping ${#NS[@]} sizes ($VERSION) -> $OUT" >&2

for n in "${NS[@]}"; do
	printf 'n=%-9s ... ' "$n" >&2
	out="$("$BIN" "$n" "$MAX_ITERS" 1.0 kdtree "$SEED")"

	tgt=$(  awk -F: '/^target points/   {gsub(/ /,"",$2); print $2}' <<<"$out")
	src=$(  awk -F: '/^source points/   {print $2}'                  <<<"$out" | awk '{print $1}')
	iters=$(awk -F: '/^iterations/      {gsub(/ /,"",$2); print $2}' <<<"$out")
	tnn=$(  awk        '/NN search/      {print $4}'                  <<<"$out")
	pct=$(  awk        '/NN search/      {gsub(/[()%]/,"",$5); print $5}' <<<"$out")
	ticp=$( awk        '/^icp total/     {print $4}'                 <<<"$out")

	nspq=$(awk -v t="$tnn" -v it="$iters" -v s="$src" \
		'BEGIN { d = it*s; printf (d>0 ? "%.3f" : "nan"), (d>0 ? t/d*1e9 : 0) }')

	echo "$n,$tgt,$src,$iters,$tnn,$ticp,$pct,$nspq" >> "$OUT"
	printf 'src=%-8s iters=%-3s t_nn=%-9s ns/query=%s\n' "$src" "$iters" "$tnn" "$nspq" >&2
done

echo "done -> $OUT" >&2
