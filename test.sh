set -o errexit

make clean 
make all 
./SimulateTraffic_cars trafficCars.csv 
# valgrind --tool=cachegrind ./SimulateTraffic_cars trafficCars.csv 
# gprof SimulateTraffic_cars gmon.out > profile.txt
