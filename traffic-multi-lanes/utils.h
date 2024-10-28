#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cuda_runtime.h>

const int NUM_LANES = 4;
constexpr int RANDOM_SEED = 47; // = 0 for fixed scenario
constexpr int NUM_CARS = (RANDOM_SEED > 0) ? 1000 : (6 * NUM_LANES); // specify #cars to randomly distribute, or use fixed scenario
const int LANE_LENGTH = 1000;
const int NUM_THREADS = 256; // Single-block implementation
const int SAFE_DISTANCE = 2;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100;

const int TEST_VERSION = 2; // implemented: V2, V3

extern int COUNT_LANE_CHANGE; // for profiling number of successful lane changes

struct CarV2 {
    int carIdx;
    int Position;
    int TargetPosition;
    int Speed;
    int TargetSpeed;
};

struct CarV3 {
    int Position;
    int TargetPosition;
    int Speed;
    int TargetSpeed;
};

struct LaneV2 {
    int numCars;
    CarV2* Cars;
};

struct LaneV3 {
    int numCars;
    int* CarIndices;
};

void sortCarsForLaneV2(LaneV2 &lane);

void sortCarIndicesForLaneV3(CarV3* &cars, LaneV3 &lane);

void initializeTrafficV2(LaneV2* &lanes);

void initializeTrafficV3(CarV3* &cars, LaneV3* &lanes);

void initializeTrafficCUDA(CarV3* &cars, int* &numCarsInLanes, int* &carIndicesInLanes);

void updateNumCars(LaneV2 &lane);

void printStepCarsV2(FILE* &fid, LaneV2* &lanes);

void printStepCarsV3(FILE* &fid, CarV3* &cars, LaneV3* &lanes);

void printStepCarsCUDA(FILE* &fid, CarV3* &cars, int* &numCarsInLanesDevice, int* &carIndicesInLanesDevice);

void execLaneChangeV2(LaneV2 &fromLane, LaneV2 &toLane, int &idxCarToMove);

bool tryLaneChangeV2(LaneV2 &lane, LaneV2 &targetLane, int &carIdx, int &laneIdx, int &targetLaneIdx);

void allCarsTryLaneChangeV2(LaneV2* &lanes, int &laneIdx);

void allCarsDriveForwardV2(LaneV2* &lanes, int &lane_index);

void execLaneChangeV3(CarV3* &cars, LaneV3 &fromLane, LaneV3 &toLane, int &laneCarIdxOfCarToMove);

bool tryLaneChangeV3(CarV3* &cars, LaneV3 &lane, LaneV3 &targetLane, int &carIdx, int &laneIdx, int &targetLaneIdx);

void allCarsTryLaneChangeV3(CarV3* &cars, LaneV3* lanes, int &laneIdx);

void allCarsDriveForwardV3(CarV3* &cars, LaneV3* &lanes, int &lane_index);

#endif