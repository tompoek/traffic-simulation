#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

void allCarsTryLaneChangeCUDA() {
    //TODO
}

__global__ 
void allCarsDriveForwardCUDA() {
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //TODO
}

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Memory allocation
    Car* carsDevice;
    checkError(cudaMalloc(&carsDevice, NUM_CARS*sizeof(*carsDevice)));

    // Initialization
    //TODO
    free(numCarsInLanes);
    free(carIndicesInLanes);

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);
        //TODO
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());

    checkError(cudaFree(carsDevice));
    return 0;
}
