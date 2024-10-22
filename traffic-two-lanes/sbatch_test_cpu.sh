#!/bin/bash
#SBATCH --job-name=simulate_traffic
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500
#SBATCH --nodes=1 # when doing MPI, distribute to multiple nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05 # time (D-HH:MM)
#SBATCH --output=SimulateTrafficTwoLanes_CPU.stdout
# #SBATCH --error=SimulateTrafficTwoLanes_CPU.stderr # enable this when debugging

set -o errexit

make clean
make all

./SimulateTrafficTwoLanes TrafficTwoLanes.csv # filename arg won't work when profiling (if printSteps are commented out)
# srun --partition=cosc3500 --account=cosc3500 ./SimulateTrafficTwoLanes TrafficTwoLanes.csv
# valgrind --tool=cachegrind ./SimulateTrafficTwoLanes 
# gprof SimulateTrafficTwoLanes gmon.out > profile.txt
