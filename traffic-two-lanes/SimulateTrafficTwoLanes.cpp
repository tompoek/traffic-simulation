#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

#include "utils.h"

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Initialization
    initializeTrafficTwoLanes();
    free(numCarsInLanes); // only for init
    free(carIndicesInLanes); // only for init
    printStep(fid); // comment out when profiling

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        // determine target position
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            cars[carIdx].TargetPosition = cars[carIdx].Position + cars[carIdx].TargetSpeed;
        }
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            // see if my front is safe
            int iAmTheFirstLeader;
            iAmTheFirstLeader = int(cars[carIdx].leaderCarIdx == -1);
            int* weAreAtTheSameLane = new int [NUM_CARS];
            std::transform(cars, cars + NUM_CARS, weAreAtTheSameLane, [carIdx](const Car& car) { 
                return int(car.laneIdx == cars[carIdx].laneIdx); /*see if we are at the same lane.*/ });
            int myFrontIsSafe;
            myFrontIsSafe = (iAmTheFirstLeader) ? 1 : int( 
                    cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition > 0 );
            // see if i can change lane
            int* distanceToMe = new int [NUM_CARS];
            std::transform(cars, cars + NUM_CARS, distanceToMe, [carIdx](const Car& car) { 
                return (car.Position - cars[carIdx].Position); /*distance will be compared to zero to see if slot is taken, or car is ahead or behind.*/ });
            int* safeToMoveHere = new int [NUM_CARS];
            std::transform(distanceToMe, distanceToMe + NUM_CARS, weAreAtTheSameLane, safeToMoveHere, [](const int& distance, const int& atTheSameLane) { 
                return int(atTheSameLane || (distance!=0)); /*if we are at the same lane, or the slot is not taken, then it's safe to move here.*/ });
            int safeToChangeLane;
            safeToChangeLane = std::accumulate(safeToMoveHere, safeToMoveHere + NUM_CARS, 
                1, std::multiplies<int>() /*if all cars say it's safe to move here, then it's safe to change lane.*/ );
            // if my front is not safe and it's safe to change lane
            if (!myFrontIsSafe && safeToChangeLane) {
                // find my closest leader car and follower car in target lane
                int closestFollowerDistance = INT_MIN; int closestLeaderDistance = INT_MAX; int closestFollowerIdx = -1; int closestLeaderIdx = -1;
                for (int carIdx2 = 0; carIdx2 < NUM_CARS; carIdx2++) {
                    int iAmBehindYou = int( distanceToMe[carIdx2] < 0 );
                    int iAmAheadOfYou = int( distanceToMe[carIdx2] > 0 );
                    if (!weAreAtTheSameLane[carIdx2] && iAmBehindYou && distanceToMe[carIdx2] > closestFollowerDistance) {
                        closestFollowerDistance = distanceToMe[carIdx2];
                        closestFollowerIdx = carIdx2;
                    } else if (!weAreAtTheSameLane[carIdx2] && iAmAheadOfYou && distanceToMe[carIdx2] < closestLeaderDistance) {
                        closestLeaderDistance = distanceToMe[carIdx2];
                        closestLeaderIdx = carIdx2;
                    }
                }
                // move myself to target lane
                cars[cars[carIdx].leaderCarIdx].followerCarIdx = /*my follower becomes my leader car's follower.*/ cars[carIdx].followerCarIdx;
                cars[cars[carIdx].followerCarIdx].leaderCarIdx = /*my leader becomes my follower car's leader.*/ cars[carIdx].leaderCarIdx;
                cars[carIdx].laneIdx = (cars[carIdx].laneIdx + 1) % 2;
                cars[carIdx].followerCarIdx = closestFollowerIdx;
                cars[carIdx].leaderCarIdx = closestLeaderIdx;
                if (closestLeaderIdx != -1) { cars[closestLeaderIdx].followerCarIdx = /*i become the closest leader car's follower.*/ carIdx; }
                if (closestFollowerIdx != -1) { cars[closestFollowerIdx].leaderCarIdx = /*i become the closest follower car's leader.*/ carIdx; }

                COUNT_LANE_CHANGE++; // for debug
            }
        }
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        // determine target position
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            cars[carIdx].TargetPosition = cars[carIdx].Position + cars[carIdx].TargetSpeed;
        }
        // see if my front is safe
        int* iAmTheFirstLeader = new int [NUM_CARS];
        int* myLeaderCarDistance = new int [NUM_CARS];
        int* myFrontIsSafe = new int [NUM_CARS];
        // std::transform(cars, cars + NUM_CARS, iAmTheFirstLeader, [](const Car& car) { return int(car.leaderCarIdx < 0); });
        // std::transform(cars, cars + NUM_CARS, iAmTheFirstLeader, myLeaderCarDistance, [](const Car& car, const int& isFirstLeader) { return isFirstLeader ? INT_MAX : cars[car.leaderCarIdx].TargetPosition - car.TargetPosition; });
        // std::transform(myLeaderCarDistance, myLeaderCarDistance + NUM_CARS, iAmTheFirstLeader, myFrontIsSafe, [](const int& distance, const int& isFirstLeader) { return int( isFirstLeader || (distance > 0) ); });
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            iAmTheFirstLeader[carIdx] = int(cars[carIdx].leaderCarIdx < 0);
            myLeaderCarDistance[carIdx] = (iAmTheFirstLeader[carIdx]) ? INT_MAX : cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition;
            myFrontIsSafe[carIdx] = int( (iAmTheFirstLeader[carIdx]) || (myLeaderCarDistance[carIdx] > 0) );
        }
        int ourFrontIsSafe;
        // ask if everyone feels safe in their front
        ourFrontIsSafe = std::accumulate(myFrontIsSafe, myFrontIsSafe + NUM_CARS, 1, std::multiplies<int>());
        while (!ourFrontIsSafe) { // as long as anyone is unsafe in their front
            // update target position
            for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
                cars[carIdx].TargetPosition = (!myFrontIsSafe[carIdx])/*if unsafe*/ * (cars[cars[carIdx].leaderCarIdx].TargetPosition - 1)/*a distance behind my leader car*/ + 
                                                (myFrontIsSafe[carIdx])/*if safe*/ * (cars[carIdx].TargetPosition)/*maintain my target position*/;
            }
            // see if my front is safe now
            // std::transform(cars, cars + NUM_CARS, iAmTheFirstLeader, [](const Car& car) { return int(car.leaderCarIdx < 0); });
            // std::transform(cars, cars + NUM_CARS, iAmTheFirstLeader, myLeaderCarDistance, [](const Car& car, const int& isFirstLeader) { return isFirstLeader ? INT_MAX : cars[car.leaderCarIdx].TargetPosition - car.TargetPosition; });
            // std::transform(myLeaderCarDistance, myLeaderCarDistance + NUM_CARS, iAmTheFirstLeader, myFrontIsSafe, [](const int& distance, const int& isFirstLeader) { return int( isFirstLeader || (distance > 0) ); });
            for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
                iAmTheFirstLeader[carIdx] = int(cars[carIdx].leaderCarIdx < 0);
                myLeaderCarDistance[carIdx] = (iAmTheFirstLeader[carIdx]) ? INT_MAX : cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition;
                myFrontIsSafe[carIdx] = int( (iAmTheFirstLeader[carIdx]) || (myLeaderCarDistance[carIdx] > 0) );
            }
            // ask if everyone feels safe in their front
            ourFrontIsSafe = std::accumulate(myFrontIsSafe, myFrontIsSafe + NUM_CARS, 1, std::multiplies<int>());
        }
        // update actual position
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            cars[carIdx].Position = cars[carIdx].TargetPosition;
            cars[carIdx].TargetSpeed = /*randomly change target speed to increase traffic dynamics.*/ (rand()%SPEED_LIMIT) + 1;
        }
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);


        printStep(fid); // comment out when profiling
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());


    free(ourFrontIsSafe);
    free(cars);
    

    return 0;
}
