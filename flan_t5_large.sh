#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=8:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=zhou.zhang@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CS6966

mkdir -p /scratch/general/vast/u1419588/huggingface_cache
export TRANSFORMER_CACHE="/scratch/general/vast/u1419588/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1419588/huggingface_cache"
OUT_DIR=/scratch/general/vast/u1419588/cs6966/assignment1/models
mkdir -p ${OUT_DIR}
python flan_t5_large.py --output_dir ${OUT_DIR}
