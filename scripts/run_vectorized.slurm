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

bash scripts/run_vectorized.sh vectorized.csv none 50 12345
