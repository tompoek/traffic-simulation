#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"

void updateNumCars(Lane &lane) {
    int num_cars = 0;
    while (lane.Cars[num_cars].Speed != 0) {
        num_cars++;
    }
    lane.numCars = num_cars;
}

// Function to sort the cars in a lane based on their position
void sortCarsForPrinting(Lane &lane) {
    // Sort the cars based on their position
    lane.SortedCars = lane.Cars;
    std::sort(lane.SortedCars, lane.SortedCars + lane.numCars, [](const Car& a, const Car& b) {
        return a.Position < b.Position;
    });
}

void printCarsInLane(FILE* &fid, Lane &lane) {
    // Sort cars in the lane first
    sortCarsForPrinting(lane);
    // Initialize lane occupancy with 0 (no car)
    int* occupancy = new int[LANE_LENGTH]();
    // Fill the lane's occupancy based on car positions and speeds
    for (int carIdx=0; carIdx < lane.numCars; ++carIdx) {
        occupancy[lane.SortedCars[carIdx].Position] = lane.SortedCars[carIdx].TargetSpeed;
    }
    // Print the lane occupancy
    for (int i = 0; i < LANE_LENGTH; i++) {
        fprintf(fid, "%d", occupancy[i]);
        if (i < LANE_LENGTH - 1) {
            fprintf(fid, ",");
        }
    }
    // Free the allocated memory for occupancy
    delete[] occupancy;
}

void printStep(FILE* &fid, Lane* lanes) {
    for (int lane_index=0; lane_index < NUM_LANES; ++lane_index) {
        if (lane_index>0) {fprintf(fid, ",");}
        printCarsInLane(fid, lanes[lane_index]);
    }
    fprintf(fid, "\n");
}

void execLaneChange(Lane &fromLane, Lane &toLane, int idxCarToMove) {
    Car &carToMove = fromLane.Cars[idxCarToMove]; // alias the car to move
    // find which index to insert the car, based on position// The car to be moved
    int insertIndex = toLane.numCars;
    for (int i=toLane.numCars-1; i>=0; --i) {
        if (toLane.Cars[i].Position <= carToMove.Position) {
            insertIndex = i;
            break;
        }
    }
    // increase the size of the target lane's Cars array to accommodate the new car
    for (int i = toLane.numCars; i > insertIndex; --i) {
        toLane.Cars[i] = toLane.Cars[i - 1];
    }
    // insert the car to the index found
    toLane.Cars[insertIndex] = carToMove;
    // assuming no lead car, drive at target speed, TBD: introduce acceleration model
    toLane.Cars[insertIndex].Speed = toLane.Cars[insertIndex].TargetSpeed;
    toLane.numCars++;
    // shift the moved car to the rightmost side
    for (int i=idxCarToMove; i<fromLane.numCars; ++i) {
        fromLane.Cars[i] = fromLane.Cars[i+1];
    }
    // delete the moved car from previous lane
    fromLane.Cars[fromLane.numCars - 1] = {0};
    fromLane.numCars--;
}

bool tryLaneChange(Lane &lane, Lane &targetLane, int &carIdx) {
    bool carHasChangedLane = false;
    bool targetLaneIsSafe = true;
    for (int ii=0; ii<targetLane.numCars; ++ii) {
        int distToCarii = targetLane.Cars[ii].Position - lane.Cars[carIdx].Position;
        if (distToCarii >=0 && distToCarii < SAFE_DISTANCE) {
            targetLaneIsSafe = false;
            break;
        }
    }
    if (targetLaneIsSafe) {
        // execute lane change
        execLaneChange(lane, targetLane, carIdx);
        carHasChangedLane = true;
    }
    return carHasChangedLane;
}

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
                        }
                        if (!carHasChangedLane && hasPreviousLane) {
                            carHasChangedLane = tryLaneChange(lane, previousLane, i);
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
