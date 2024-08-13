#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cstdio>
#include <random>

const int LANE_LENGTH = 50;
const int NUM_LANES = 4;
constexpr int RANDOM_SEED = 47; // = 0 for fixed scenario
constexpr int NUM_CARS = (RANDOM_SEED > 0) ? 40 : (6 * NUM_LANES); // specify #cars to randomly distribute, or use fixed scenario
const int SAFE_DISTANCE = 2;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100000;

struct Car {
    int carIdx;
    int Position;
    int TargetPosition;
    int Speed;
    int TargetSpeed;
};

struct Lane {
    int numCars;
    Car* Cars;
    Car* SortedCars;
};

void printHeaderSpaceOccupancy(FILE* &fid);

void sortCars(Lane &lane);

void initializeTraffic(Lane* &lanes);

void updateNumCars(Lane &lane);

void sortCarsForPrinting(Lane &lane);

void printStepCars(FILE* &fid, Lane* &lanes);

void printLaneOccupancy(FILE* &fid, Lane &lane);

void printStepSpaceOccupancy(FILE* &fid, Lane* lanes);

void execLaneChange(Lane &fromLane, Lane &toLane, int &idxCarToMove);

bool tryLaneChange(Lane &lane, Lane &targetLane, int &carIdx, int &laneIdx, int &targetLaneIdx);

#endif