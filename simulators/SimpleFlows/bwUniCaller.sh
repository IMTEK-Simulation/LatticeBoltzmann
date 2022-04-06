#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --partition=dev_multiple
#SBATCH --ntasks-per-node=25
#SBATCH --mail-user=christoph.mos.studium@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slidingLidMPI.out
#SBATCH --error=slidingLidMPI.err
echo "Loading Pythona module and mpi module"
module load devel/python/3.10.0_gnu_11.1
module load mpi/openmpi/4.1
module list
startexe="mpirun --bind-to core --map-by core -report-bindings python3 ./slidingLidMPI.py"
exec $startexe
