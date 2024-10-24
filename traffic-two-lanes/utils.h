#ifndef UTILS_H
#define UTILS_H

const int RANDOM_SEED = 47;
const int NUM_CARS = 500;
const int LANE_LENGTH = 1.5 * NUM_CARS;
const int NUM_THREADS = 4; // must be exponential of 2
const int NUM_BLOCKS = (NUM_CARS + NUM_THREADS - 1) / NUM_THREADS;
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
extern Car* carsTemp;

extern int* numCarsInLanes; // only used for init
extern int* carIndicesInLanes; // only used for init

void initializeTrafficTwoLanes();

void printStep(FILE* &fid);

bool checkFirstLeaderLastFollower(int laneIdx);

#endif