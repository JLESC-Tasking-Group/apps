#!/bin/bash
#SBATCH --job-name=ovniemu
#SBATCH --chdir=.
#SBATCH --output=./ovniemu.out
#SBATCH --error=./ovniemu.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --qos=gp_bsccs

ovniemu $1
