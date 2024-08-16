#include "utils.h"

void sortCarsForLaneV2(LaneV2 &lane) {
    // Sort the cars based on their position
    std::sort(lane.Cars, lane.Cars + lane.numCars, [](const CarV2& a, const CarV2& b) {
        return a.Position < b.Position;
    });
}

void sortCarIndicesForLaneV3(CarV3* &cars, LaneV3 &lane) {
    // Sort the cars based on their position
    std::sort(lane.CarIndices, lane.CarIndices + lane.numCars, [&cars](int a, int b) {
        return cars[a].Position < cars[b].Position;
    });
}

void initializeTrafficV2(LaneV2* &lanes) {
    for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
        lanes[lane_index].numCars = 0;
        lanes[lane_index].Cars = static_cast<CarV2*>(malloc(sizeof(CarV2) * LANE_LENGTH));
    }
    if (RANDOM_SEED > 0) { // Random>> Distribute NUM_CARS cars randomly across all lanes
        int lane_idx, pos_idx;
        std::vector<int> indices(NUM_LANES*LANE_LENGTH);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(RANDOM_SEED); // random seed
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int carIdx=0; carIdx<NUM_CARS; ++carIdx) {
            lane_idx = indices[carIdx]/LANE_LENGTH;
            pos_idx = indices[carIdx]%LANE_LENGTH;
            // fill a car according to lane_idx and pos_idx, assigning random speed up to SPEED_LIMIT
            lanes[lane_idx].numCars++;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].carIdx = carIdx;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Position = pos_idx;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed = (rand()%SPEED_LIMIT) + 1;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].TargetSpeed = lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed;
        }
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            sortCarsForLaneV2(lanes[lane_index]); // sort cars for physical lead-follow relations
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
        // already sorted, no need to sort
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
            // already sorted, no need to sort
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
                // already sorted, no need to sort
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
                    // already sorted, no need to sort
                }
            }
        }
    }
}

void initializeTrafficV3(CarV3* &cars, LaneV3* &lanes) {
    for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
        lanes[lane_index].numCars = 0;
        lanes[lane_index].CarIndices = static_cast<int*>(malloc(sizeof(int) * LANE_LENGTH));
    }
    if (RANDOM_SEED > 0) { // Random>> Distribute NUM_CARS cars randomly across all lanes
        int lane_idx, pos_idx;
        std::vector<int> indices(NUM_LANES*LANE_LENGTH);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(RANDOM_SEED); // random seed
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int carIdx=0; carIdx<NUM_CARS; ++carIdx) {
            lane_idx = indices[carIdx]/LANE_LENGTH;
            pos_idx = indices[carIdx]%LANE_LENGTH;
            // fill a car according to lane_idx and pos_idx, assigning random speed up to SPEED_LIMIT
            lanes[lane_idx].numCars++;
            lanes[lane_idx].CarIndices[lanes[lane_idx].numCars-1] = carIdx;
            cars[carIdx].Position = pos_idx;
            cars[carIdx].Speed = (rand()%SPEED_LIMIT) + 1;
            cars[carIdx].TargetSpeed = cars[carIdx].Speed;
        }
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            // sort cars for physical lead-follow relations
            sortCarIndicesForLaneV3(cars, lanes[lane_index]);
        }
    } else { // Fixed>> curated initial conditions:
        lanes[0].numCars = 6;
        lanes[0].CarIndices[0] = 0;
        lanes[0].CarIndices[1] = 1;
        lanes[0].CarIndices[2] = 2;
        lanes[0].CarIndices[3] = 3;
        lanes[0].CarIndices[4] = 4;
        lanes[0].CarIndices[5] = 5;
        cars[0].Position = 0; // Car A
        cars[0].Speed = 3;
        cars[0].TargetSpeed = cars[0].Speed;
        cars[1].Position = 1; // Car B
        cars[1].Speed = 3;
        cars[1].TargetSpeed = cars[1].Speed;
        cars[2].Position = 2; // Car C
        cars[2].Speed = 2;
        cars[2].TargetSpeed = cars[2].Speed;
        cars[3].Position = 20;
        cars[3].Speed = 2;
        cars[3].TargetSpeed = cars[3].Speed;
        cars[4].Position = 21;
        cars[4].Speed = 1;
        cars[4].TargetSpeed = cars[4].Speed;
        cars[5].Position = 22;
        cars[5].Speed = 1;
        cars[5].TargetSpeed = cars[5].Speed;
        // already sorted, no need to sort
        if (NUM_LANES > 1) {
            lanes[1].numCars = 6;
            lanes[1].CarIndices[0] = 6;
            lanes[1].CarIndices[1] = 7;
            lanes[1].CarIndices[2] = 8;
            lanes[1].CarIndices[3] = 9;
            lanes[1].CarIndices[4] = 10;
            lanes[1].CarIndices[5] = 11;
            cars[6].Position = 10;
            cars[6].Speed = 4;
            cars[6].TargetSpeed = cars[6].Speed;
            cars[7].Position = 11;
            cars[7].Speed = 2;
            cars[7].TargetSpeed = cars[7].Speed;
            cars[8].Position = 12;
            cars[8].Speed = 2;
            cars[8].TargetSpeed = cars[8].Speed;
            cars[9].Position = 30;
            cars[9].Speed = 3;
            cars[9].TargetSpeed = cars[9].Speed;
            cars[10].Position = 31;
            cars[10].Speed = 1;
            cars[10].TargetSpeed = cars[10].Speed;
            cars[11].Position = 32;
            cars[11].Speed = 1;
            cars[11].TargetSpeed = cars[11].Speed;
            // already sorted, no need to sort
            if (NUM_LANES > 2) {
                lanes[2].numCars = 6;
                lanes[2].CarIndices[0] = 12;
                lanes[2].CarIndices[1] = 13;
                lanes[2].CarIndices[2] = 14;
                lanes[2].CarIndices[3] = 15;
                lanes[2].CarIndices[4] = 16;
                lanes[2].CarIndices[5] = 17;
                cars[12].Position = 32;
                cars[12].Speed = 4;
                cars[12].TargetSpeed = cars[12].Speed;
                cars[13].Position = 35;
                cars[13].Speed = 4;
                cars[13].TargetSpeed = cars[13].Speed;
                cars[14].Position = 38;
                cars[14].Speed = 4;
                cars[14].TargetSpeed = cars[14].Speed;
                cars[15].Position = 41;
                cars[15].Speed = 4;
                cars[15].TargetSpeed = cars[15].Speed;
                cars[16].Position = 44;
                cars[16].Speed = 4;
                cars[16].TargetSpeed = cars[16].Speed;
                cars[17].Position = 47;
                cars[17].Speed = 4;
                cars[17].TargetSpeed = cars[17].Speed;
                // already sorted, no need to sort
                if (NUM_LANES > 3) {
                    lanes[3].numCars = 6;
                    lanes[3].CarIndices[0] = 18;
                    lanes[3].CarIndices[1] = 19;
                    lanes[3].CarIndices[2] = 20;
                    lanes[3].CarIndices[3] = 21;
                    lanes[3].CarIndices[4] = 22;
                    lanes[3].CarIndices[5] = 23;
                    cars[18].Position = 36;
                    cars[18].Speed = 3;
                    cars[18].TargetSpeed = cars[18].Speed;
                    cars[19].Position = 38;
                    cars[19].Speed = 3;
                    cars[19].TargetSpeed = cars[19].Speed;
                    cars[20].Position = 40;
                    cars[20].Speed = 3;
                    cars[20].TargetSpeed = cars[20].Speed;
                    cars[21].Position = 42;
                    cars[21].Speed = 3;
                    cars[21].TargetSpeed = cars[21].Speed;
                    cars[22].Position = 44;
                    cars[22].Speed = 3;
                    cars[22].TargetSpeed = cars[22].Speed;
                    cars[23].Position = 46;
                    cars[23].Speed = 3;
                    cars[23].TargetSpeed = cars[23].Speed;
                    // already sorted, no need to sort
                }
            }
        }
    }
}

void updateNumCars(LaneV2 &lane) {
    int num_cars = 0;
    while (lane.Cars[num_cars].Speed != 0) {
        num_cars++;
    }
    lane.numCars = num_cars;
}

void printStepCarsV2(FILE* &fid, LaneV2* &lanes) {
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

void printStepCarsV3(FILE* &fid, CarV3* &cars, LaneV3* &lanes) {
    for (int carIdx = 0; carIdx < NUM_CARS; ++carIdx) {
        if (carIdx>0) fprintf(fid, ",");
        bool carFound = false;
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            for (int j=0; j<lanes[laneIdx].numCars; ++j) {
                if (carIdx == lanes[laneIdx].CarIndices[j]) {
                    fprintf(fid, "%d,%d", laneIdx, cars[carIdx].Position);
                    carFound = true;
                    break;
                }
            }
            if (carFound) {break;}
        }
    }
    fprintf(fid, "\n");
}

void execLaneChangeV2(LaneV2 &fromLane, LaneV2 &toLane, int &laneCarIdxOfCarToMove) {
    CarV2 &carToMove = fromLane.Cars[laneCarIdxOfCarToMove]; // alias the car to move
    // find which index to insert the car, based on position// The car to be moved
    int toLaneInsertIndex = 0;
    for (int i=toLane.numCars-1; i>=0; --i) {
        if (toLane.Cars[i].Position < carToMove.Position) { // == case should not happen, because lane must be safe!
            toLaneInsertIndex = i;
            break;
        }
    }
    // shift all cars in target lane after toLaneInsertIndex
    for (int i = toLane.numCars; i > toLaneInsertIndex; --i) {
        toLane.Cars[i] = toLane.Cars[i - 1];
    }
    // insert the car to the index found
    toLane.Cars[toLaneInsertIndex] = carToMove;
    // assuming no lead car, drive at target speed, TBD: introduce acceleration model
    toLane.Cars[toLaneInsertIndex].Speed = toLane.Cars[toLaneInsertIndex].TargetSpeed;
    toLane.numCars++;
    // shift the moved car to the rightmost side
    for (int i=laneCarIdxOfCarToMove; i<fromLane.numCars; ++i) {
        fromLane.Cars[i] = fromLane.Cars[i+1];
    }
    // delete the moved car from previous lane
    fromLane.Cars[fromLane.numCars - 1] = {0};
    fromLane.numCars--;
}

bool tryLaneChangeV2(LaneV2 &lane, LaneV2 &targetLane, int &laneCarIdx, int &laneIdx, int &targetLaneIdx) {
    bool carHasChangedLane = false;
    bool targetLaneIsSafe = true;
    for (int ii=0; ii<targetLane.numCars; ++ii) {
        int distToCarii = ((targetLane.Cars[ii].Position - lane.Cars[laneCarIdx].Position) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
        if (distToCarii < SAFE_DISTANCE) {
            targetLaneIsSafe = false;
            break;
        }
    }
    if (targetLaneIsSafe) {
        // execute lane change
        printf("Car %d changes from Lane %d to %d\n", lane.Cars[laneCarIdx].carIdx, laneIdx, targetLaneIdx);
        execLaneChangeV2(lane, targetLane, laneCarIdx);
        carHasChangedLane = true;
    }
    return carHasChangedLane;
}

void allCarsTryLaneChangeV2(LaneV2* &lanes, int &laneIdx) {
    LaneV2 &lane = lanes[laneIdx];
    bool hasLeadCar = false;
    bool hasNextLane = true;
    int nextLaneIdx = laneIdx + 1;
    if (laneIdx == NUM_LANES - 1) {hasNextLane = false; nextLaneIdx = 0;}
    LaneV2 &nextLane = lanes[nextLaneIdx];
    bool hasPreviousLane = true;
    int previousLaneIdx = laneIdx - 1;
    if (laneIdx == 0) {hasPreviousLane = false; previousLaneIdx = NUM_LANES - 1;}
    LaneV2 &previousLane = lanes[previousLaneIdx];
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
                    carHasChangedLane = tryLaneChangeV2(lane, nextLane, i, laneIdx, nextLaneIdx);
                }
                if (!carHasChangedLane && hasPreviousLane) {
                    carHasChangedLane = tryLaneChangeV2(lane, previousLane, i, laneIdx, previousLaneIdx);
                }
            }
        }
    }
}

void allCarsDriveForwardV2(LaneV2* &lanes, int &lane_index) {
    LaneV2 &lane = lanes[lane_index];
    int &numCars = lane.numCars;

    // Determine target positions
    for (int i = 0; i < numCars; ++i) {
        lane.Cars[i].TargetPosition = lane.Cars[i].Position + lane.Cars[i].Speed;
    }
    for (int i = numCars - 1; i > 0; i--) {
        int num_collisions = 0;
        for (int j = i - 1; j >= 0; j--) {
            if (lane.Cars[j].TargetPosition >= lane.Cars[i].TargetPosition) { // ASSUMPTION: speeds never exceeds LANE_LENGTH
                // Collision detected, move car j as close as possible without colliding
                num_collisions++;
                lane.Cars[j].TargetPosition = lane.Cars[i].TargetPosition - num_collisions;
                lane.Cars[j].Speed = lane.Cars[i].Speed; // then adjust car j's speed to lead car i's speed
            }
        }
    }
    // Update positions after collisions are resolved
    for (int i = 0; i < numCars; ++i) {
        lane.Cars[i].Position = ((lane.Cars[i].TargetPosition % LANE_LENGTH) + LANE_LENGTH) % LANE_LENGTH;
    }
}

void execLaneChangeV3(CarV3* &cars, LaneV3 &fromLane, LaneV3 &toLane, int &laneCarIdxOfCarToMove) {
    int &carIdx = fromLane.CarIndices[laneCarIdxOfCarToMove];
    CarV3 &carToMove = cars[carIdx]; // alias the car to move
    // find which index to insert the car, based on position// The car to be moved
    int toLaneInsertIndex = 0;
    for (int i=toLane.numCars-1; i>=0; --i) {
        if (cars[toLane.CarIndices[i]].Position < carToMove.Position) { // == case should not happen, because lane must be safe!
            toLaneInsertIndex = i;
            break;
        }
    }
    // shift all car indices in target lane after toLaneInsertIndex
    for (int i = toLane.numCars; i > toLaneInsertIndex; --i) {
        toLane.CarIndices[i] = toLane.CarIndices[i - 1]; // THIS IS THE EXPECTED OPTMIZATION OF V3 vs V2
    }
    // insert the car index to the index found
    toLane.CarIndices[toLaneInsertIndex] = carIdx; // THIS IS THE EXPECTED OPTMIZATION OF V3 vs V2
    // assuming no lead car, drive at target speed, TBD: introduce acceleration model
    carToMove.Speed = carToMove.TargetSpeed;
    toLane.numCars++;
    // shift the moved car's index to the rightmost side
    for (int i=laneCarIdxOfCarToMove; i<fromLane.numCars; ++i) {
        fromLane.CarIndices[i] = fromLane.CarIndices[i+1];
    }
    // delete the moved car from previous lane
    fromLane.CarIndices[fromLane.numCars - 1] = 0;
    fromLane.numCars--;
}

bool tryLaneChangeV3(CarV3* &cars, LaneV3 &lane, LaneV3 &targetLane, int &laneCarIdx, int &laneIdx, int &targetLaneIdx) {
    bool carHasChangedLane = false;
    bool targetLaneIsSafe = true;
    for (int ii=0; ii<targetLane.numCars; ++ii) {
        int distToCarii = ((cars[targetLane.CarIndices[ii]].Position - cars[lane.CarIndices[laneCarIdx]].Position) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
        if (distToCarii < SAFE_DISTANCE) {
            targetLaneIsSafe = false;
            break;
        }
    }
    if (targetLaneIsSafe) {
        // execute lane change
        printf("Car %d changes from Lane %d to %d\n", lane.CarIndices[laneCarIdx], laneIdx, targetLaneIdx);
        execLaneChangeV3(cars, lane, targetLane, laneCarIdx);
        carHasChangedLane = true;
    }
    return carHasChangedLane;
}

void allCarsTryLaneChangeV3(CarV3* &cars, LaneV3* lanes, int &laneIdx) {
    LaneV3 &lane = lanes[laneIdx];
    bool hasLeadCar = false;
    bool hasNextLane = true;
    int nextLaneIdx = laneIdx + 1;
    if (laneIdx == NUM_LANES - 1) {hasNextLane = false; nextLaneIdx = 0;}
    LaneV3 &nextLane = lanes[nextLaneIdx];
    bool hasPreviousLane = true;
    int previousLaneIdx = laneIdx - 1;
    if (laneIdx == 0) {hasPreviousLane = false; previousLaneIdx = NUM_LANES - 1;}
    LaneV3 &previousLane = lanes[previousLaneIdx];
    if (lane.numCars > 1) {
        for (int i = lane.numCars-2; i >= 0; --i) {
            hasLeadCar = false;
            int &carPos = cars[lane.CarIndices[i]].Position;
            for (int j = i+1; j < lane.numCars; ++j) {
                // detect lead car in the current lane
                int distToCarj = ((cars[lane.CarIndices[j]].Position - carPos) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
                if (distToCarj < SAFE_DISTANCE) {
                    hasLeadCar = true;
                    break;
                }
            }
            if (hasLeadCar) {
                bool carHasChangedLane = false;
                // detect cars in the target lane
                if (hasNextLane) {
                    carHasChangedLane = tryLaneChangeV3(cars, lane, nextLane, i, laneIdx, nextLaneIdx);
                }
                if (!carHasChangedLane && hasPreviousLane) {
                    carHasChangedLane = tryLaneChangeV3(cars, lane, previousLane, i, laneIdx, previousLaneIdx);
                }
            }
        }
    }
}

void allCarsDriveForwardV3(CarV3* &cars, LaneV3* &lanes, int &lane_index) {
    LaneV3 &lane = lanes[lane_index];
    int &laneNumCars = lane.numCars;

    // Determine target positions
    for (int i = 0; i < laneNumCars; ++i) {
        cars[lane.CarIndices[i]].TargetPosition = cars[lane.CarIndices[i]].Position + cars[lane.CarIndices[i]].Speed;
    }
    for (int i = laneNumCars - 1; i > 0; i--) {
        int num_collisions = 0;
        for (int j = i - 1; j >= 0; j--) {
            if (cars[lane.CarIndices[j]].TargetPosition >= cars[lane.CarIndices[i]].TargetPosition) { // ASSUMPTION: speeds never exceeds LANE_LENGTH
                // Collision detected, move car j as close as possible without colliding
                num_collisions++;
                cars[lane.CarIndices[j]].TargetPosition = cars[lane.CarIndices[i]].TargetPosition - num_collisions;
                cars[lane.CarIndices[j]].Speed = cars[lane.CarIndices[i]].Speed; // then adjust car j's speed to lead car i's speed
            }
        }
    }
    // Update positions after collisions are resolved
    for (int i = 0; i < laneNumCars; ++i) {
        cars[lane.CarIndices[i]].Position = ((cars[lane.CarIndices[i]].TargetPosition % LANE_LENGTH) + LANE_LENGTH) % LANE_LENGTH;
    }
}

