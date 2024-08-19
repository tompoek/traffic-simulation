#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"

int COUNT_LANE_CHANGE = 0; // for profiling number of successful lane changes

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout;

    // Memory allocation
    LaneV2* lanesV2 = static_cast<LaneV2*>(malloc(sizeof(LaneV2) * NUM_LANES));
    CarV3* carsV3 = static_cast<CarV3*>(malloc(sizeof(CarV3)* NUM_CARS));
    LaneV3* lanesV3 = static_cast<LaneV3*>(malloc(sizeof(LaneV3) * NUM_LANES));

    // Initialization
    if (TEST_VERSION == 2) {
        initializeTrafficV2(lanesV2);
        printStepCarsV2(fid, lanesV2);
    } else {
        initializeTrafficV3(carsV3, lanesV3);
        printStepCarsV3(fid, carsV3, lanesV3);
    }

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        // printf("@ Step %d\n", step);
        // Try Lane change
        for (int laneIdx=0; laneIdx < NUM_LANES; ++laneIdx) {
            start_clock = std::chrono::high_resolution_clock::now();
            if (TEST_VERSION == 2) {
                allCarsTryLaneChangeV2(lanesV2, laneIdx);
            } else {
                allCarsTryLaneChangeV3(carsV3, lanesV3, laneIdx);
            }
            microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        }

        // All cars drive forward, must resolve collisions before updating positions
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            start_clock = std::chrono::high_resolution_clock::now();
            if (TEST_VERSION == 2) {
                allCarsDriveForwardV2(lanesV2, lane_index);
            } else {
                allCarsDriveForwardV3(carsV3, lanesV3, lane_index);
            }
            microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        }

        if (TEST_VERSION == 2) {
            printStepCarsV2(fid, lanesV2);
        } else {
            printStepCarsV3(fid, carsV3, lanesV3);
        }
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());

    free(lanesV2);
    free(carsV3);
    free(lanesV3);

    return 0;
}
