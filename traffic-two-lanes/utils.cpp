#include <iostream>
#include <cstdio>
#include <random>
#include <algorithm>

#include "utils.h"

int* ourFrontIsSafe = static_cast<int*>(malloc(sizeof(*ourFrontIsSafe) * 2)); // for two-lanes implementation
Car* cars = static_cast<Car*>(malloc(sizeof(*cars) * NUM_CARS));

int COUNT_LANE_CHANGE = 0; // for profiling number of successful lane changes
int* numCarsInLanes = static_cast<int*>(malloc(sizeof(int) * 2)); // for two-lanes implementation
int* carIndicesInLanes = static_cast<int*>(malloc(sizeof(int) * 2 * LANE_LENGTH)); // for two-lanes implementation


void initializeTrafficTwoLanes() {
    ourFrontIsSafe[0] = 1;
    ourFrontIsSafe[1] = 1;
    numCarsInLanes[0] = 0;
    numCarsInLanes[1] = 0;
    // Random>> Distribute NUM_CARS cars randomly across all lanes
    int laneIdx, position;
    std::vector<int> indices(2*LANE_LENGTH);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
    std::mt19937 gen(RANDOM_SEED); // random seed
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int carIdx=0; carIdx<NUM_CARS; ++carIdx) {
        laneIdx = indices[carIdx]/LANE_LENGTH;
        position = /*update: the farther ahead, the smaller index*/ LANE_LENGTH-1 - (indices[carIdx]%LANE_LENGTH);
        // fill a car according to laneIdx and position, assigning random speed up to SPEED_LIMIT
        numCarsInLanes[laneIdx]++;
        cars[carIdx].laneIdx = laneIdx;
        carIndicesInLanes[laneIdx*LANE_LENGTH + numCarsInLanes[laneIdx] - 1] = carIdx; // Link carIdx to the last element in corresponding lane_idx of carIndicesInLanes matrix
        cars[carIdx].Position = position;
        cars[carIdx].TargetSpeed = (rand()%SPEED_LIMIT) + 1;
        cars[carIdx].TargetPosition = (position + cars[carIdx].TargetSpeed) % LANE_LENGTH; // doesn't matter for now
    }
    for (int laneIdx = 0; laneIdx < 2; laneIdx++) {
        // Sort the cars based on their position
        int* tempCarIndices = new int[LANE_LENGTH];
        // load car indices of one lane to temp
        for (int i = 0; i < numCarsInLanes[laneIdx]; i++) {
            tempCarIndices[i] = carIndicesInLanes[laneIdx*LANE_LENGTH + i];
        }
        // sort the temp array
        Car* &tempCars = cars;
        std::sort(tempCarIndices, tempCarIndices + numCarsInLanes[laneIdx], [tempCars](int a, int b) {
            return tempCars[a].Position < tempCars[b].Position;
        });
        // store temp array back to car indices in that lane
        for (int i = 0; i < numCarsInLanes[laneIdx]; i++) {
            carIndicesInLanes[laneIdx*LANE_LENGTH + i] = tempCarIndices[i];
        }
    }
    // using numCarsInLanes and carIndicesInLanes, set leadcarIdx and followerCarIdx for each car
    for (int laneIdx = 0; laneIdx < 2; laneIdx++) {
        for (int laneCarIdx = 0; laneCarIdx < numCarsInLanes[laneIdx]; laneCarIdx++) {
            int followerLaneCarIdx = laneCarIdx - 1;
            int leaderLaneCarIdx = laneCarIdx + 1;
            int &carIdx = carIndicesInLanes[laneIdx*LANE_LENGTH + laneCarIdx];
            int followerCarIdx = (followerLaneCarIdx > 0) ? carIndicesInLanes[laneIdx*LANE_LENGTH + followerLaneCarIdx] : -1; // -1: no follower
            int leaderCarIdx = (leaderLaneCarIdx < numCarsInLanes[laneIdx]) ? carIndicesInLanes[laneIdx*LANE_LENGTH + leaderLaneCarIdx] : -1; // -1: no leader
            cars[carIdx].leaderCarIdx = leaderCarIdx;
            cars[carIdx].followerCarIdx = followerCarIdx;
            // cars[leaderCarIdx].followerCarIdx = carIdx; // duplicate assignment
            // cars[followerCarIdx].leaderCarIdx = carIdx; // duplicate assignment
        }
    }
}

void printStep(FILE* &fid) {
    for (int carIdx = 0; carIdx < NUM_CARS; ++carIdx) {
        if (carIdx>0) fprintf(fid, ",");
        fprintf(fid, "%d,%d", cars[carIdx].laneIdx, (cars[carIdx].Position%LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH);
    }
    fprintf(fid, "\n");
}