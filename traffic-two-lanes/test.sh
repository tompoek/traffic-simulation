set -o errexit

make clean 
make all 

rm -f TrafficTwoLanes.csv
./SimulateTrafficTwoLanes TrafficTwoLanes.csv # filename arg won't work when profiling (if printSteps are commented out)
# valgrind --tool=cachegrind ./SimulateTrafficTwoLanes 
# gprof SimulateTrafficTwoLanes gmon.out > profile.txt

# rm -f TrafficTwoLanes_CUDA.csv
# ncu -fo report_CUDA.ncu-rep ./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv # filename arg won't work when profiling (if printSteps are commented out)

# rm -f TrafficTwoLanes_thrust.csv
# ncu -fo report_thrust.ncu-rep ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv # filename arg won't work when profiling (if printSteps are commented out)
