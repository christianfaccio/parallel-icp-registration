#!/bin/bash
#SBATCH -J icp_vec_bf
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -o vec_v3_bf_%j.out -e vec_v3_bf_%j.err

AFF=$(taskset -cp $$ | awk -F': ' '{print $2}')

taskset -c "$AFF" perf stat -C "$AFF" -e cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/vectorized/icp_vectorized_v3 500000 50 1.0 brute

taskset -c "$AFF" perf record -C "$AFF" -g --call-graph dwarf -F 999 ./bin/vectorized/icp_vectorized_v3 500000 50 1.0 brute

perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > vec_v3_bf_flame.svg
