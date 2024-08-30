set -o errexit

make clean 
make all 
./SimulateTraffic_cars trafficCars.csv # filename arg won't work when profiling (if printSteps are commented out)
# valgrind --tool=cachegrind ./SimulateTraffic_cars 
# gprof SimulateTraffic_cars gmon.out > profile.txt
