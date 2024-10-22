set -o errexit

make clean 
make all 

# ./SimulateTrafficTwoLanes TrafficTwoLanes.csv # filename arg won't work when profiling (if printSteps are commented out)
# valgrind --tool=cachegrind ./SimulateTrafficTwoLanes 
# gprof SimulateTrafficTwoLanes gmon.out > profile.txt

# ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv # filename arg won't work when profiling (if printSteps are commented out)
# ncu -fo report_thrust.ncu-rep ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv

./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv # filename arg won't work when profiling (if printSteps are commented out)
# ncu -fo report_CUDA.ncu-rep ./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv
