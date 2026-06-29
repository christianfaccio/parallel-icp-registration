#!/bin/bash
#SBATCH -J icp_cuda
#SBATCH -p boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -o cuda_%j.out -e cuda_%j.err

# Which versioned GPU backend to build and sweep (v0, v1, ...).
VERSION="${VERSION:-v1}"

bash scripts/run_cuda.sh "out/cuda/cuda_$VERSION.csv" "$VERSION" 50 12345
