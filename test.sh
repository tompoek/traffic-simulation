make clean && 
make all && 
./SimulateTraffic_cars trafficSpaceOccupancy.csv && 
gprof SimulateTraffic_cars gmon.out > profile.txt
