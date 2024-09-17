#include <iostream>
#include <chrono>
#include <curand_kernel.h>

#include "utils.h"

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Initialization
    initializeTrafficTwoLanes();
    free(numCarsInLanes); // only for init
    free(carIndicesInLanes); // only for init
    printStep(fid); // comment out when profiling

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        //TODO
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        //TODO
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);


        printStep(fid); // comment out when profiling
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());


    free(cars);


    return 0;
}