set -o errexit

make clean 
make all 
./SimulateTrafficMultiLanes TrafficMultiLanes.csv # filename arg won't work when profiling (if printSteps are commented out)
# valgrind --tool=cachegrind ./SimulateTrafficMultiLanes TrafficMultiLanes.csv
# gprof SimulateTrafficMultiLanes gmon.out > profile.txt
