#ifndef UTILS_H
#define UTILS_H

const int LANE_LENGTH = 2000;
const int NUM_LANES = 2;
const int RANDOM_SEED = 47;
const int NUM_CARS = 3000;
const int NUM_THREADS = 256;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100;

extern int COUNT_LANE_CHANGE; // for debugging number of successful lane changes

struct Car {
    int laneIdx;
    int leaderCarIdx;
    int followerCarIdx;
    int Position;
    int TargetPosition;
    int TargetSpeed;
};

extern Car* cars;

extern int* numCarsInLanes; // only used for init
extern int* carIndicesInLanes; // only used for init

void initializeTrafficTwoLanes();

void printStep(FILE* &fid);

#endif