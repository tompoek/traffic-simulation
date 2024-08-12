#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cstdio>
#include <random>

const int LANE_LENGTH = 50;
const int NUM_LANES = 4;
const int NUM_CARS = 40; // for randomly distributed cars
// const int NUM_CARS = 6 * NUM_LANES; // for fixed scenario
const int SAFE_DISTANCE = 2;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100000;

struct Car {
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

void printHeader(FILE* &fid);

void sortCars(Lane &lane);

void initializeTraffic(Lane* &lanes);

#endif