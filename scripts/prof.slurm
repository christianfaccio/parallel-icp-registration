#!/bin/bash
#SBATCH -J icp_vectorized
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -o serial_%j.out -e serial_%j.err

AFF=$(taskset -cp $$ | awk -F': ' '{print $2}')

taskset -c "$AFF" perf stat -C "$AFF" -e cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/vectorized/icp_vectorized_v3 100000

taskset -c "$AFF" perf record -C "$AFF" -g --call-graph dwarf -F 999 ./bin/vectorized/icp_vectorized_v3 100000

perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > vec_v3_flame.svg
