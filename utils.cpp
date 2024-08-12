#include "utils.h"

void printHeader(FILE* &fid) {
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
    int randomSeed = 47; // 0 represents fixed scenario
    if (randomSeed > 0) { // Random>> Distribute NUM_CARS cars randomly across all lanes
        int lane_idx, pos_idx;
        std::vector<int> indices(NUM_LANES*LANE_LENGTH);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(randomSeed); // random seed
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int j=0; j<NUM_CARS; ++j) {
            lane_idx = indices[j]/LANE_LENGTH;
            pos_idx = indices[j]%LANE_LENGTH;
            // fill a car according to lane_idx and pos_idx, assigning random speed up to SPEED_LIMIT
            lanes[lane_idx].numCars++;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Position = pos_idx;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed = (rand()%SPEED_LIMIT) + 1;
            lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].TargetSpeed = lanes[lane_idx].Cars[lanes[lane_idx].numCars-1].Speed;
        }
        for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
            sortCars(lanes[lane_index]);
        }
    } else { // Fixed>> curated initial conditions:
        lanes[0].numCars = 6;
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