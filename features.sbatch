#!/bin/bash
#SBATCH --job-name=psgbind-features
#SBATCH --mem=25G

#SBATCH --time=0:20:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jones657@msu.edu

#SBATCH --nodes=1 
#SBATCH --cpus-per-task=25
##SBATCH --gres=gpu:v100:1 
#SBATCH --output=./output/%x-%j.out 
source /mnt/home/jones657/.bashrc
conda activate py36
cd /mnt/home/jones657/Documents/psg_bind
python feature_analysis.py
scontrol show job $SLURM_JOB_ID
