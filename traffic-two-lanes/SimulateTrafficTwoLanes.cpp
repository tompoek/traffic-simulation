#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstring>

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
        // /*copy cars values to carsTemp*/memcpy(carsTemp, cars, NUM_CARS*sizeof(*carsTemp));
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
                // carsTemp[cars[carIdx].leaderCarIdx].followerCarIdx = /*my follower becomes my leader car's follower.*/ cars[carIdx].followerCarIdx;
                // carsTemp[cars[carIdx].followerCarIdx].leaderCarIdx = /*my leader becomes my follower car's leader.*/ cars[carIdx].leaderCarIdx;
                // carsTemp[carIdx].laneIdx = (cars[carIdx].laneIdx + 1) % 2;
                // carsTemp[carIdx].followerCarIdx = closestFollowerIdx;
                // carsTemp[carIdx].leaderCarIdx = closestLeaderIdx;
                // if (closestLeaderIdx != -1) { carsTemp[closestLeaderIdx].followerCarIdx = /*i become the closest leader car's follower.*/ carIdx; }
                // if (closestFollowerIdx != -1) { carsTemp[closestFollowerIdx].leaderCarIdx = /*i become the closest follower car's leader.*/ carIdx; }
                cars[cars[carIdx].leaderCarIdx].followerCarIdx = /*my follower becomes my leader car's follower.*/ cars[carIdx].followerCarIdx;
                cars[cars[carIdx].followerCarIdx].leaderCarIdx = /*my leader becomes my follower car's leader.*/ cars[carIdx].leaderCarIdx;
                cars[carIdx].laneIdx = (cars[carIdx].laneIdx + 1) % 2;
                cars[carIdx].followerCarIdx = closestFollowerIdx;
                cars[carIdx].leaderCarIdx = closestLeaderIdx;
                if (closestLeaderIdx != -1) { cars[closestLeaderIdx].followerCarIdx = /*i become the closest leader car's follower.*/ carIdx; }
                if (closestFollowerIdx != -1) { cars[closestFollowerIdx].leaderCarIdx = /*i become the closest follower car's leader.*/ carIdx; }

                COUNT_LANE_CHANGE++; // for debug
                // /*DEBUG*/printf("Car[%d] just changed from Lane%d to Lane%d\n", carIdx, cars[carIdx].laneIdx, carsTemp[carIdx].laneIdx);
            }
        }
        // /*copy carsTemp values back to cars*/std::swap(cars, carsTemp);
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        // determine target position
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            cars[carIdx].TargetPosition = cars[carIdx].Position + cars[carIdx].TargetSpeed;
        }

        // resolve collisions if any
        for (int laneIdx = 0; laneIdx < 2; laneIdx++) {
            // /*DEBUG*/ bool checked = checkFirstLeaderLastFollower(laneIdx); if (!checked) return -1;
            // find the first leader car index
            int* iAmThisLanesFirstLeader = new int [NUM_CARS];
            std::transform(cars, cars + NUM_CARS, iAmThisLanesFirstLeader, [laneIdx](const Car& car) { return int(car.leaderCarIdx < 0 && car.laneIdx == laneIdx); });
            int leaderCarIdx = std::distance(iAmThisLanesFirstLeader, std::find(iAmThisLanesFirstLeader, iAmThisLanesFirstLeader + NUM_CARS, 1));
            int followerCarIdx = cars[leaderCarIdx].followerCarIdx;
            int* iAmAtCurrentLane = new int [NUM_CARS];
            std::transform(cars, cars + NUM_CARS, iAmAtCurrentLane, [laneIdx](const Car& car) { return int(car.laneIdx == laneIdx); });
            int numCarsAtCurrentLane = std::accumulate(iAmAtCurrentLane, iAmAtCurrentLane + NUM_CARS, 0, std::plus<int>());
            for (int laneCarIdx = 0; laneCarIdx < numCarsAtCurrentLane; laneCarIdx++) {
                if (followerCarIdx == -1) break;
                if (cars[followerCarIdx].TargetPosition >= cars[leaderCarIdx].TargetPosition /*my follower would hit me*/) {
                    cars[followerCarIdx].TargetPosition = cars[leaderCarIdx].TargetPosition - 1/*tell them to move a distance behind me*/;
                    // /*DEBUG*/printf("@Lane%d: Car %d tells Car %d to move back to %d.\n", laneIdx, leaderCarIdx, followerCarIdx, cars[followerCarIdx].TargetPosition);
                }
                leaderCarIdx = followerCarIdx;
                followerCarIdx = cars[leaderCarIdx].followerCarIdx;
            }
        }

        // update actual position
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            cars[carIdx].Position = cars[carIdx].TargetPosition;
            // cars[carIdx].TargetSpeed = /*randomly change target speed to increase traffic dynamics.*/ (rand()%SPEED_LIMIT) + 1;
        }
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);


        printStep(fid); // comment out when profiling
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());


    free(cars);
    

    return 0;
}
