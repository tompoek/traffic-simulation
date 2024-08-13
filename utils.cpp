#include "utils.h"

void printHeaderSpaceOccupancy(FILE* &fid) {
    for (int i=0; i<NUM_LANES; ++i) {
        if (i>0) {fprintf(fid,",");}
        for (int j=0; j<LANE_LENGTH; ++j) {
            if (j>0) {fprintf(fid,",");}
            fprintf(fid,"Lane_%d_Space_%d", i, j);
        }
    }
    fprintf(fid,"\n");
}

void sortCars(Lane &lane) {
    // Sort the cars based on their position
    std::sort(lane.Cars, lane.Cars + lane.numCars, [](const Car& a, const Car& b) {
        return a.Position < b.Position;
    });
}

void initializeTraffic(Lane* &lanes) {
    for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
        lanes[lane_index].numCars = 0;
        lanes[lane_index].Cars = static_cast<Car*>(malloc(sizeof(Car) * LANE_LENGTH));
    }
    if (RANDOM_SEED > 0) { // Random>> Distribute NUM_CARS cars randomly across all lanes
        int lane_idx, pos_idx;
        std::vector<int> indices(NUM_LANES*LANE_LENGTH);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(RANDOM_SEED); // random seed
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int j=0; j<NUM_CARS; ++j) {
            lane_idx = indices[j]/LANE_LENGTH;
            pos_idx = indices[j]%LANE_LENGTH;
            // fill a car according to lane_idx and pos_idx, assigning random speed up to SPEED_LIMIT
            lanes[lane_idx].numCars++;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].carIdx = j;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Position = pos_idx;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed = (rand()%SPEED_LIMIT) + 1;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].TargetSpeed = lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed;
        }
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            sortCars(lanes[lane_index]); // sort cars for physical lead-follow relations
        }
    } else { // Fixed>> curated initial conditions:
        lanes[0].numCars = 6;
        lanes[0].Cars[0].carIdx = 0;
        lanes[0].Cars[1].carIdx = 1;
        lanes[0].Cars[2].carIdx = 2;
        lanes[0].Cars[3].carIdx = 3;
        lanes[0].Cars[4].carIdx = 4;
        lanes[0].Cars[5].carIdx = 5;
        lanes[0].Cars[0].Position = 0; // Car A
        lanes[0].Cars[0].Speed = 3;
        lanes[0].Cars[0].TargetSpeed = lanes[0].Cars[0].Speed;
        lanes[0].Cars[1].Position = 1; // Car B
        lanes[0].Cars[1].Speed = 3;
        lanes[0].Cars[1].TargetSpeed = lanes[0].Cars[1].Speed;
        lanes[0].Cars[2].Position = 2; // Car C
        lanes[0].Cars[2].Speed = 2;
        lanes[0].Cars[2].TargetSpeed = lanes[0].Cars[2].Speed;
        lanes[0].Cars[3].Position = 20;
        lanes[0].Cars[3].Speed = 2;
        lanes[0].Cars[3].TargetSpeed = lanes[0].Cars[3].Speed;
        lanes[0].Cars[4].Position = 21;
        lanes[0].Cars[4].Speed = 1;
        lanes[0].Cars[4].TargetSpeed = lanes[0].Cars[4].Speed;
        lanes[0].Cars[5].Position = 22;
        lanes[0].Cars[5].Speed = 1;
        lanes[0].Cars[5].TargetSpeed = lanes[0].Cars[5].Speed;
        // sortCars(lanes[0]);
        if (NUM_LANES > 1) {
            lanes[1].numCars = 6;
            lanes[1].Cars[0].carIdx = 6;
            lanes[1].Cars[1].carIdx = 7;
            lanes[1].Cars[2].carIdx = 8;
            lanes[1].Cars[3].carIdx = 9;
            lanes[1].Cars[4].carIdx = 10;
            lanes[1].Cars[5].carIdx = 11;
            lanes[1].Cars[0].Position = 10;
            lanes[1].Cars[0].Speed = 4;
            lanes[1].Cars[0].TargetSpeed = lanes[1].Cars[0].Speed;
            lanes[1].Cars[1].Position = 11;
            lanes[1].Cars[1].Speed = 2;
            lanes[1].Cars[1].TargetSpeed = lanes[1].Cars[1].Speed;
            lanes[1].Cars[2].Position = 12;
            lanes[1].Cars[2].Speed = 2;
            lanes[1].Cars[2].TargetSpeed = lanes[1].Cars[2].Speed;
            lanes[1].Cars[3].Position = 30;
            lanes[1].Cars[3].Speed = 3;
            lanes[1].Cars[3].TargetSpeed = lanes[1].Cars[3].Speed;
            lanes[1].Cars[4].Position = 31;
            lanes[1].Cars[4].Speed = 1;
            lanes[1].Cars[4].TargetSpeed = lanes[1].Cars[4].Speed;
            lanes[1].Cars[5].Position = 32;
            lanes[1].Cars[5].Speed = 1;
            lanes[1].Cars[5].TargetSpeed = lanes[1].Cars[5].Speed;
            // sortCars(lanes[1]);
            if (NUM_LANES > 2) {
                lanes[2].numCars = 6;
                lanes[2].Cars[0].carIdx = 12;
                lanes[2].Cars[1].carIdx = 13;
                lanes[2].Cars[2].carIdx = 14;
                lanes[2].Cars[3].carIdx = 15;
                lanes[2].Cars[4].carIdx = 16;
                lanes[2].Cars[5].carIdx = 17;
                lanes[2].Cars[0].Position = 32;
                lanes[2].Cars[0].Speed = 4;
                lanes[2].Cars[0].TargetSpeed = lanes[2].Cars[0].Speed;
                lanes[2].Cars[1].Position = 35;
                lanes[2].Cars[1].Speed = 4;
                lanes[2].Cars[1].TargetSpeed = lanes[2].Cars[1].Speed;
                lanes[2].Cars[2].Position = 38;
                lanes[2].Cars[2].Speed = 4;
                lanes[2].Cars[2].TargetSpeed = lanes[2].Cars[2].Speed;
                lanes[2].Cars[3].Position = 41;
                lanes[2].Cars[3].Speed = 4;
                lanes[2].Cars[3].TargetSpeed = lanes[2].Cars[3].Speed;
                lanes[2].Cars[4].Position = 44;
                lanes[2].Cars[4].Speed = 4;
                lanes[2].Cars[4].TargetSpeed = lanes[2].Cars[4].Speed;
                lanes[2].Cars[5].Position = 47;
                lanes[2].Cars[5].Speed = 4;
                lanes[2].Cars[5].TargetSpeed = lanes[2].Cars[5].Speed;
                // sortCars(lanes[2]);
                if (NUM_LANES > 3) {
                    lanes[3].numCars = 6;
                    lanes[3].Cars[0].carIdx = 18;
                    lanes[3].Cars[1].carIdx = 19;
                    lanes[3].Cars[2].carIdx = 20;
                    lanes[3].Cars[3].carIdx = 21;
                    lanes[3].Cars[4].carIdx = 22;
                    lanes[3].Cars[5].carIdx = 23;
                    lanes[3].Cars[0].Position = 36;
                    lanes[3].Cars[0].Speed = 3;
                    lanes[3].Cars[0].TargetSpeed = lanes[3].Cars[0].Speed;
                    lanes[3].Cars[1].Position = 38;
                    lanes[3].Cars[1].Speed = 3;
                    lanes[3].Cars[1].TargetSpeed = lanes[3].Cars[1].Speed;
                    lanes[3].Cars[2].Position = 40;
                    lanes[3].Cars[2].Speed = 3;
                    lanes[3].Cars[2].TargetSpeed = lanes[3].Cars[2].Speed;
                    lanes[3].Cars[3].Position = 42;
                    lanes[3].Cars[3].Speed = 3;
                    lanes[3].Cars[3].TargetSpeed = lanes[3].Cars[3].Speed;
                    lanes[3].Cars[4].Position = 44;
                    lanes[3].Cars[4].Speed = 3;
                    lanes[3].Cars[4].TargetSpeed = lanes[3].Cars[4].Speed;
                    lanes[3].Cars[5].Position = 46;
                    lanes[3].Cars[5].Speed = 3;
                    lanes[3].Cars[5].TargetSpeed = lanes[3].Cars[5].Speed;
                    // sortCars(lanes[3]);
                }
            }
        }
    }
}

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

void printStepCars(FILE* &fid, Lane* &lanes) {
    for (int carIdx = 0; carIdx < NUM_CARS; ++carIdx) {
        if (carIdx>0) fprintf(fid, ",");
        bool carFound = false;
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            for (int j=0; j<lanes[laneIdx].numCars; ++j) {
                if (carIdx == lanes[laneIdx].Cars[j].carIdx) {
                    fprintf(fid, "%d,%d", laneIdx, lanes[laneIdx].Cars[j].Position);
                    carFound = true;
                    break;
                }
            }
            if (carFound) {break;}
        }
    }
    fprintf(fid, "\n");
}

void printLaneOccupancy(FILE* &fid, Lane &lane) {
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

void printStepSpaceOccupancy(FILE* &fid, Lane* lanes) {
    for (int lane_index=0; lane_index < NUM_LANES; ++lane_index) {
        if (lane_index>0) {fprintf(fid, ",");}
        printLaneOccupancy(fid, lanes[lane_index]);
    }
    fprintf(fid, "\n");
}

void execLaneChange(Lane &fromLane, Lane &toLane, int &idxCarToMove) {
    Car &carToMove = fromLane.Cars[idxCarToMove]; // alias the car to move
    // find which index to insert the car, based on position// The car to be moved
    int insertIndex = 0;
    for (int i=toLane.numCars-1; i>=0; --i) {
        if (toLane.Cars[i].Position < carToMove.Position) { // == case should not happen, because lane must be safe!
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

bool tryLaneChange(Lane &lane, Lane &targetLane, int &carIdx, int &laneIdx, int &targetLaneIdx) {
    bool carHasChangedLane = false;
    bool targetLaneIsSafe = true;
    for (int ii=0; ii<targetLane.numCars; ++ii) {
        int distToCarii = ((targetLane.Cars[ii].Position - lane.Cars[carIdx].Position) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
        if (distToCarii < SAFE_DISTANCE) {
            targetLaneIsSafe = false;
            break;
        }
    }
    if (targetLaneIsSafe) {
        // execute lane change
        printf("Car %d changes from Lane %d to %d\n", carIdx, laneIdx, targetLaneIdx);
        execLaneChange(lane, targetLane, carIdx);
        carHasChangedLane = true;
    }
    return carHasChangedLane;
}

