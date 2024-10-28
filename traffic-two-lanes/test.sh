set -o errexit

make clean 
make all 

echo -e "\n>>> Running CPU code >>>"
./SimulateTrafficTwoLanes TrafficTwoLanes.csv # filename arg won't work when profiling (if printSteps are commented out)
# valgrind --tool=cachegrind ./SimulateTrafficTwoLanes SimulateTrafficTwoLanes.csv
# gprof SimulateTrafficTwoLanes gmon.out > profile.txt

echo -e "\n>>> Running GPU code with thrust >>>"
./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv # filename arg won't work when profiling (if printSteps are commented out)
# ncu -fo report_thrust.ncu-rep ./SimulateTrafficTwoLanes_thrust TrafficTwoLanes_thrust.csv

echo -e "\n>>> Running GPU code with CUDA manual malloc >>>"
./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv # filename arg won't work when profiling (if printSteps are commented out)
# ncu -fo report_CUDA.ncu-rep ./SimulateTrafficTwoLanes_CUDA TrafficTwoLanes_CUDA.csv
