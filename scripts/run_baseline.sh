#!/usr/bin/env bash
#
# sweep_baseline.sh -- serial NN-cost vs cache-size sweep for the ICP baseline.
#
# Runs bin/baseline/icp_baseline over a geometric sweep of n that brackets the
# L1/L2/L3 cache boundaries, and emits a CSV whose key column is the AMORTIZED
# time per nearest-neighbour query:
#
#     ns_per_query = t_nn / (iters * source_points) * 1e9
#
# That normalization removes the rising total-work trend so the cache knees show
# up as steps. Plot ns_per_query vs n on a log-x axis and draw vertical lines at
# the n* markers printed below.
#
# Cache markers (Leonardo BOOSTER, Xeon 8358 Ice Lake, 28 B/target-point):
#     L1d  48 KB  -> n* ~ 1755
#     L2  1.25 MB -> n* ~ 46811
#     L3   48 MB  -> n* ~ 1797559
# (Note: the boost_usr_prod nodes use Ice Lake, NOT the DCGP Sapphire Rapids.
#  If you ever get a DCGP allocation, the markers become 1755 / 74898 / 3932160.)
#
# Usage:
#     make baseline
#     tools/sweep_baseline.sh [output.csv] [core] [max_iters] [seed]
#
# Defaults: out=baseline_sweep.csv, core=1, max_iters=50, seed=12345.
# Pin to an isolated core for stable counters; on Leonardo run inside an
# salloc'd DCGP compute node, never the login node.

set -euo pipefail

# Force '.' as decimal separator so awk's printf doesn't emit locale commas
# (which would corrupt the CSV).
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$HERE/bin/baseline/icp_baseline"

OUT="${1:-baseline_sweep.csv}"
CORE="${2:-auto}"   # "auto" = pin to the first CPU in our cgroup; or an explicit id; "none" = no pin
MAX_ITERS="${3:-50}"
SEED="${4:-12345}"

if [[ ! -x "$BIN" ]]; then
	echo "error: $BIN not found. Build it first with: make baseline" >&2
	exit 1
fi

# Pin to one core for stable timings. Under SLURM the job's cpuset rarely
# includes physical core 0, so hardcoding -c 0 fails ("Invalid argument").
# Default "auto": derive the first CPU actually allowed for this process.
# If you allocate --cpus-per-task=1 the cgroup already pins you; use CORE=none.
PIN=()
if [[ "$CORE" != "none" ]] && command -v taskset >/dev/null 2>&1; then
	if [[ "$CORE" == "auto" ]]; then
		CORE=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //' | cut -d, -f1 | cut -d- -f1)
	fi
	if [[ -n "$CORE" ]] && taskset -c "$CORE" true 2>/dev/null; then
		PIN=(taskset -c "$CORE")
	else
		echo "warning: cannot pin to core '$CORE'; running without taskset" >&2
	fi
fi

# Geometric sweep with the three Ice Lake (Booster) cache n* values as markers.
NS=(
	500 1000 1755 3000 6000 12000 25000 46811 80000
	150000 300000 500000 1000000 1797559 3000000 5000000
)

echo "n,target_points,source_points,iters,t_nn_s,t_icp_s,nn_pct,ns_per_query" > "$OUT"
echo "# sweeping ${#NS[@]} sizes -> $OUT" >&2

for n in "${NS[@]}"; do
	printf 'n=%-9s ... ' "$n" >&2
	out="$("${PIN[@]}" "$BIN" "$n" "$MAX_ITERS" 1.0 kdtree "$SEED")"

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
