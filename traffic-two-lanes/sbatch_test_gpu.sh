#!/bin/bash
#SBATCH --job-name=simulate_traffic
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500
#SBATCH --nodes=1 # when doing MPI, distribute to multiple nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0-00:05 # time (D-HH:MM)
#SBATCH --output=SimulateTrafficTwoLanes_GPU.stdout
# #SBATCH --error=SimulateTrafficTwoLanes_GPU.stderr # enable this when debugging

set -o errexit

make clean
make all

# ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv # filename arg won't work when profiling (if printSteps are commented out)
# srun --partition=cosc3500 --account=cosc3500 --gpus=1 ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv
# ncu -fo report_thrust.ncu-rep ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv

./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv # filename arg won't work when profiling (if printSteps are commented out)
# srun --partition=cosc3500 --account=cosc3500 --gpus=1 ./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv
# ncu -fo report_CUDA.ncu-rep ./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv