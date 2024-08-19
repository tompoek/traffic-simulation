#!/bin/bash
#SBATCH --job-name=simulate_traffic
#SBATCH --partition=largecpu
#SBATCH --nodes=1 # when doing MPI, distribute to multiple nodes
#SBATCH --ntasks=1 # when doing MPI, specify several tasks
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # single thread, if multiple threads, > 1
# #SBATCH --mem-per-cpu=1M # memory (MB)
#SBATCH --time=0-00:05 # time (D-HH:MM)
#SBATCH --output=SimulateTraffic_largecpu.stdout
#SBATCH --error=SimulateTraffic_largecpu.stderr

set -o errexit

make clean 
make all 
./SimulateTraffic_cars trafficCars.csv 
gprof SimulateTraffic_cars gmon.out > profile_largecpu.txt
