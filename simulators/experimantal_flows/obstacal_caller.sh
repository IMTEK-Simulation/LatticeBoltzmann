#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=single
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=christoph.mos.studium@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slidingLidMPI.out
#SBATCH --error=slidingLidMPI.err
echo "Loading Python module and mpi module"
module load devel/python/3.10.0_gnu_11.1
module load mpi/openmpi/4.1
module list
startexe="mpirun --bind-to core --map-by core -report-bindings python3 ./obstacle_canal.py"
exec $startexe
