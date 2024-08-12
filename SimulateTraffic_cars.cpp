#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_tryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_driveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout;
    // printHeader(fid);

    // Initialization
    Lane* lanes = static_cast<Lane*>(malloc(sizeof(Lane) * NUM_LANES));
    initializeTraffic(lanes);

    printStep(fid, lanes);

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);
        // Try Lane change
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            start_clock = std::chrono::high_resolution_clock::now();
            Lane &lane = lanes[laneIdx];
            bool hasLeadCar = false;
            bool hasNextLane = true;
            if (laneIdx == NUM_LANES - 1) {hasNextLane = false;}
            Lane &nextLane = lanes[(laneIdx + 1) % NUM_LANES];
            bool hasPreviousLane = true;
            if (laneIdx == 0) {hasPreviousLane = false;}
            Lane &previousLane = lanes[((laneIdx - 1) % NUM_LANES + NUM_LANES) % NUM_LANES];
            if (lane.numCars > 1) {
                for (int i = lane.numCars-2; i >= 0; --i) {
                    hasLeadCar = false;
                    int &carPos = lane.Cars[i].Position;
                    for (int j = i+1; j < lane.numCars; ++j) {
                        // detect lead car in the current lane
                        int distToCarj = ((lane.Cars[j].Position - carPos) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
                        if (distToCarj < SAFE_DISTANCE) {
                            hasLeadCar = true;
                            break;
                        }
                    }
                    if (hasLeadCar) {
                        bool carHasChangedLane = false;
                        // detect cars in the target lane
                        if (hasNextLane) {
                            carHasChangedLane = tryLaneChange(lane, nextLane, i);
                            if (carHasChangedLane) {printf("Car %d has moved from Lane %d to its Right Lane\n", i, laneIdx);}
                        }
                        if (!carHasChangedLane && hasPreviousLane) {
                            carHasChangedLane = tryLaneChange(lane, previousLane, i);
                            if (carHasChangedLane) {printf("Car %d has moved from Lane %d to its Left Lane\n", i, laneIdx);}
                        }
                    }
                }
            }
            microsecs_tryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        }

        // All cars drive forward, must resolve collisions before updating positions
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            start_clock = std::chrono::high_resolution_clock::now();
            Lane &lane = lanes[lane_index];
            int &numCars = lane.numCars;

            // Determine target positions
            for (int i = 0; i < numCars; ++i) {
                lane.Cars[i].TargetPosition = lane.Cars[i].Position + lane.Cars[i].Speed;
            }
            for (int i = numCars - 1; i > 0; i--) {
                for (int j = i - 1; j >= 0; j--) {
                    if (lane.Cars[j].TargetPosition >= lane.Cars[i].TargetPosition) { // ASSUMPTION: speeds never exceeds LANE_LENGTH
                        // Collision detected, move car j as close as possible without colliding
                        lane.Cars[j].TargetPosition = lane.Cars[i].TargetPosition - 1;
                        lane.Cars[j].Speed = lane.Cars[i].Speed; // then adjust car j's speed to lead car i's speed
                    }
                }
            }
            // Update positions after collisions are resolved
            for (int i = 0; i < numCars; ++i) {
                lane.Cars[i].Position = ((lane.Cars[i].TargetPosition % LANE_LENGTH) + LANE_LENGTH) % LANE_LENGTH;
            }
            microsecs_driveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        }

        printStep(fid, lanes);
    }
    printf("Num Steps: %d, Num Lanes: %d\n", NUM_STEPS, NUM_LANES);
    printf("Cumulative microseconds of tryLaneChange = %ld us\n", microsecs_tryLaneChange.count());
    printf("Cumulative microseconds of driveForward = %ld us\n", microsecs_driveForward.count());

    free(lanes);

    return 0;
}
