#!/bin/bash
#
#SBATCH --job-name=of_test
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=03-12:00       # Runtime in D-HH:MM

export CUDA_VISIBLE_DEVICES=2
./test.sh 2 of-48 150001
