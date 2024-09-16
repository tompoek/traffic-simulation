#include <iostream>
#include <vector>
#include <chrono>
#include <curand_kernel.h>

#include "utils.h"

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

__global__ 
void determineTargetPositionCUDA(Car* cars) {
    int thrIdx = threadIdx.x;
    int numThreads = blockDim.x;
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        cars[carIdx].TargetPosition = cars[carIdx].Position + cars[carIdx].TargetSpeed;
    }
}

__global__ 
void tryLaneChangeCUDA(Car* cars, int* countLaneChange) {
    int thrIdx = threadIdx.x;
    int numThreads = blockDim.x;
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        // see if my front is safe
        int iAmTheFirstLeader;
        iAmTheFirstLeader = int(cars[carIdx].leaderCarIdx == -1);
        __shared__ /*per block shared memory*/ int weAreAtTheSameLane[NUM_CARS];
        for (int carIdx2 = 0; carIdx2 < NUM_CARS; carIdx2++) {
            weAreAtTheSameLane[carIdx2] = int(cars[carIdx2].laneIdx == cars[carIdx].laneIdx);
        }
        int myFrontIsSafe;
        myFrontIsSafe = (iAmTheFirstLeader) ? 1 : int( 
                cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition > 0 );
        // see if i can change lane
        __shared__ int distanceToMe[NUM_CARS];
        for (int carIdx2 = 0; carIdx2 < NUM_CARS; carIdx2++) {
            distanceToMe[carIdx2] = cars[carIdx2].Position - cars[carIdx].Position;
        }
        __shared__ int safeToMoveHere[NUM_CARS];
        for (int carIdx2 = 0; carIdx2 < NUM_CARS; carIdx2++) {
            safeToMoveHere[carIdx2] = int(weAreAtTheSameLane[carIdx2] || (distanceToMe[carIdx2]!=0));
        }
        int safeToChangeLane = 1; //TODO: optimize reduction
        for (int carIdx2 = 0; carIdx2 < NUM_CARS; carIdx2++) {
            safeToChangeLane *= safeToMoveHere[carIdx2];
        }
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
                // // ISSUE: atomicMin / atomicMax would cause CUDA error: 719 : unspecified launch failure
                // if (!weAreAtTheSameLane[carIdx2] && iAmBehindYou) {
                //     atomicMax(&closestFollowerDistance, distanceToMe[carIdx2]);
                //     if (closestFollowerDistance == distanceToMe[carIdx2]) {
                //         closestFollowerIdx = carIdx2;
                //     }
                // } else if (!weAreAtTheSameLane[carIdx2] && iAmAheadOfYou) {
                //     atomicMin(&closestLeaderDistance, distanceToMe[carIdx2]);
                //     if (closestLeaderDistance == distanceToMe[carIdx2]) {
                //         closestLeaderIdx = carIdx2;
                //     }
                // }
            }
            // move myself to target lane
            cars[cars[carIdx].leaderCarIdx].followerCarIdx = /*my follower becomes my leader car's follower.*/ cars[carIdx].followerCarIdx;
            cars[cars[carIdx].followerCarIdx].leaderCarIdx = /*my leader becomes my follower car's leader.*/ cars[carIdx].leaderCarIdx;
            cars[carIdx].laneIdx = (cars[carIdx].laneIdx + 1) % 2;
            cars[carIdx].followerCarIdx = closestFollowerIdx;
            cars[carIdx].leaderCarIdx = closestLeaderIdx;
            if (closestLeaderIdx != -1) { cars[closestLeaderIdx].followerCarIdx = /*i become the closest leader car's follower.*/ carIdx; }
            if (closestFollowerIdx != -1) { cars[closestFollowerIdx].leaderCarIdx = /*i become the closest follower car's leader.*/ carIdx; }

            countLaneChange[0]++; // for debug
            printf("Car[%d] just changed lane!!\n", carIdx);
        }
    }
}

__global__ 
void resolveCollisionsCUDA(Car* cars, int* ourFrontIsSafe) {
    int thrIdx = threadIdx.x;
    int numThreads = blockDim.x;
    // see if my front is safe
    __shared__ int iAmTheFirstLeader[NUM_CARS];
    __shared__ int myLeaderCarDistance[NUM_CARS];
    __shared__ int myFrontIsSafe[NUM_CARS];
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        iAmTheFirstLeader[carIdx] = int(cars[carIdx].leaderCarIdx < 0);
        myLeaderCarDistance[carIdx] = (iAmTheFirstLeader[carIdx]) ? INT_MAX : cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition;
        myFrontIsSafe[carIdx] = int( (iAmTheFirstLeader[carIdx]) || (myLeaderCarDistance[carIdx] > 0) );
    }
    // ask if everyone feels safe in their front
    ourFrontIsSafe[thrIdx] = 1;
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        ourFrontIsSafe[thrIdx] *= myFrontIsSafe[carIdx];
    }
    for (int i = numThreads/2; i > 0; i /= 2) {
        __syncthreads(); // must wait for all threads to reach here
        if (thrIdx < i) {
            ourFrontIsSafe[thrIdx] *= ourFrontIsSafe[thrIdx + i] /*only thread 0 gets correct result*/;
        }
    }

    //DEBUG >>>
    __shared__ int numLoops;
    numLoops = 0;
    //DEBUG >>>
    
    while (!ourFrontIsSafe[0] /*only thread 0 gets correct result*/) { // as long as anyone is unsafe in their front
    //DEBUG >>>
    numLoops++;
    if (numLoops>100) {printf("Error: too many loops!"); break;}
    //DEBUG >>>
        // update target position
        for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
            cars[carIdx].TargetPosition = (!myFrontIsSafe[carIdx])/*if unsafe*/ * (cars[cars[carIdx].leaderCarIdx].TargetPosition - 1)/*a distance behind my leader car*/ + 
                                            (myFrontIsSafe[carIdx])/*if safe*/ * (cars[carIdx].TargetPosition)/*maintain my target position*/;
        }
        // see if my front is safe now
        for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
            iAmTheFirstLeader[carIdx] = int(cars[carIdx].leaderCarIdx < 0);
            myLeaderCarDistance[carIdx] = (iAmTheFirstLeader[carIdx]) ? INT_MAX : cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition;
            myFrontIsSafe[carIdx] = int( (iAmTheFirstLeader[carIdx]) || (myLeaderCarDistance[carIdx] > 0) );
        }
        // ask if everyone feels safe in their front
        ourFrontIsSafe[thrIdx] = 1;
        for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
            ourFrontIsSafe[thrIdx] *= myFrontIsSafe[carIdx];
        }
        for (int i = numThreads/2; i > 0; i /= 2) {
            __syncthreads(); // must wait for all threads to reach here
            if (thrIdx < i) {
                ourFrontIsSafe[thrIdx] *= ourFrontIsSafe[thrIdx + i] /*only thread 0 gets correct result*/;
            }
        }
    }
}

__global__ 
void updateActualPositionCUDA(Car* cars) {
    int thrIdx = threadIdx.x;
    int numThreads = blockDim.x;
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        cars[carIdx].Position = cars[carIdx].TargetPosition;
        // curandState state;
        // // Q: does random state get updated when different threads call curand()? 
        // // A: Yes, the curandState does get updated when different threads call curand().
        // // However, CUDA implementation will yield different results from CPU.
        // curand_init(clock64(), carIdx, 0, &state);
        // cars[carIdx].TargetSpeed = /*randomly change target speed to increase traffic dynamics.*/ (curand(&state)%SPEED_LIMIT) + 1;
    }
}

int main(int argc, char** argv) {
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Memory allocation
    Car* carsDevice;
    checkError(cudaMalloc(&carsDevice, NUM_CARS*sizeof(*carsDevice)));
    int* ourFrontIsSafe;
    checkError(cudaMalloc(&ourFrontIsSafe, NUM_THREADS*sizeof(*ourFrontIsSafe)));
    int* countLaneChangeDevice;
    checkError(cudaMalloc(&countLaneChangeDevice, sizeof(*countLaneChangeDevice)));
    checkError(cudaMemcpy(countLaneChangeDevice, &COUNT_LANE_CHANGE, sizeof(*countLaneChangeDevice), cudaMemcpyHostToDevice));

    // Initialization
    initializeTrafficTwoLanes();
    free(numCarsInLanes);
    free(carIndicesInLanes);
    printStep(fid);
    checkError(cudaMemcpy(carsDevice, cars, NUM_CARS*sizeof(*carsDevice), cudaMemcpyHostToDevice));

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        determineTargetPositionCUDA<<<1, NUM_THREADS>>>(carsDevice);
        tryLaneChangeCUDA<<<1, NUM_THREADS>>>(carsDevice, countLaneChangeDevice);
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        
        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        determineTargetPositionCUDA<<<1, NUM_THREADS>>>(carsDevice);
        resolveCollisionsCUDA<<<1, NUM_THREADS>>>(carsDevice, ourFrontIsSafe);
        updateActualPositionCUDA<<<1, NUM_THREADS>>>(carsDevice);
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        
        checkError(cudaMemcpy(cars, carsDevice, NUM_CARS*sizeof(*carsDevice), cudaMemcpyDeviceToHost));
        printStep(fid);
    }
    checkError(cudaMemcpy(&COUNT_LANE_CHANGE, countLaneChangeDevice, sizeof(*countLaneChangeDevice), cudaMemcpyDeviceToHost));
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());

    checkError(cudaFree(carsDevice));
    checkError(cudaFree(ourFrontIsSafe));
    checkError(cudaFree(countLaneChangeDevice));
    return 0;
}
