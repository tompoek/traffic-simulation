#include <iostream>
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
    int thrIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    for (int carIdx = thrIdx; carIdx < NUM_CARS; carIdx += numThreads) {
        cars[carIdx].TargetPosition = cars[carIdx].Position + cars[carIdx].TargetSpeed;
    }
}

__global__ 
void eachCarTryLaneChangeCUDA(Car* cars, int* carIdxDevice, int* countLaneChange) {
    int &carIdx = carIdxDevice[0];
    int thrIdx = threadIdx.x;
    int numThreads = blockDim.x;
        // see if my front is safe
        int iAmTheFirstLeader;
        iAmTheFirstLeader = int(cars[carIdx].leaderCarIdx == -1);
        __shared__ /*per block shared memory*/ int weAreAtTheSameLane[NUM_CARS];
        for (int carIdx2 = thrIdx; carIdx2 < NUM_CARS; carIdx2 += numThreads) {
            weAreAtTheSameLane[carIdx2] = int(cars[carIdx2].laneIdx == cars[carIdx].laneIdx);
        }
        __shared__ int myFrontIsSafe;
        myFrontIsSafe = (iAmTheFirstLeader) ? 1 : int( 
                cars[cars[carIdx].leaderCarIdx].TargetPosition - cars[carIdx].TargetPosition > 0 );
        // see if i can change lane
        __shared__ int distanceToMe[NUM_CARS];
        for (int carIdx2 = thrIdx; carIdx2 < NUM_CARS; carIdx2 += numThreads) {
            distanceToMe[carIdx2] = cars[carIdx2].Position - cars[carIdx].Position;
        }
        __shared__ int safeToMoveHere[NUM_CARS];
        for (int carIdx2 = thrIdx; carIdx2 < NUM_CARS; carIdx2 += numThreads) {
            safeToMoveHere[carIdx2] = int(weAreAtTheSameLane[carIdx2] || (distanceToMe[carIdx2]!=0));
        }
        
        extern __shared__ int safeToChangeLane[];
        safeToChangeLane[thrIdx] = 1;
        for (int carIdx2 = thrIdx; carIdx2 < NUM_CARS; carIdx2 += numThreads) {
            safeToChangeLane[thrIdx] *= safeToMoveHere[carIdx2];
        }
        for (int i = numThreads/2; i > 0; i /= 2) {
            __syncthreads(); // must wait for all threads to reach here
            if (thrIdx < i) {
                safeToChangeLane[thrIdx] *= safeToChangeLane[thrIdx + i] /*only thread 0 gets correct result*/;
            }
        }
        // if my front is not safe and it's safe to change lane
        if (!myFrontIsSafe && safeToChangeLane[0]) {
            // find my closest leader car and follower car in target lane
            int closestFollowerDistance = INT_MIN; int closestLeaderDistance = INT_MAX; int closestFollowerIdx = -1; int closestLeaderIdx = -1;
            for (int carIdx2 = thrIdx; carIdx2 < NUM_CARS; carIdx2 += numThreads) {
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
            countLaneChange[0]++; // for debug
            // /*DEBUG*/if (thrIdx == 0) printf("Car[%d] just changed lane!!\n", carIdx);
        }
}

__global__ 
void tryLaneChangeCUDA(Car* cars, int* countLaneChange) {
    int thrIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
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
        int safeToChangeLane = 1; // reduction could be optimized if we thread inner loop, but threading inner loop requires frequent memcpy making it slower
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
            // /*DEBUG*/printf("Car[%d] just changed from Lane%d to Lane%d\n", carIdx, (cars[carIdx].laneIdx + 1) % 2, cars[carIdx].laneIdx);
        }
    }
}

__global__
void resolveCollisionsPerLaneCUDA(Car* cars) {
    int laneIdx = threadIdx.x;
    // find the first leader car index
    int leaderCarIdx = -1;
    for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
        if (cars[carIdx].leaderCarIdx < 0 && cars[carIdx].laneIdx == laneIdx) {
            leaderCarIdx = carIdx;
            break;
        }
    }
    if (leaderCarIdx != -1) {
        int followerCarIdx = cars[leaderCarIdx].followerCarIdx;
        int numCarsAtCurrentLane = 0;
        for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) {
            if (cars[carIdx].laneIdx == laneIdx) {numCarsAtCurrentLane++;}
        }
        for (int laneCarIdx = 0; laneCarIdx < numCarsAtCurrentLane; laneCarIdx++) {
            if (followerCarIdx == -1) break;
            if (cars[followerCarIdx].TargetPosition >= cars[leaderCarIdx].TargetPosition /*my follower would hit me*/) {
                cars[followerCarIdx].TargetPosition = cars[leaderCarIdx].TargetPosition - 1/*tell them to move a distance behind me*/;
                // /*DEBUG*/printf("@Lane%d: Car %d tells Car %d to move back to %d.\n", laneIdx, leaderCarIdx, followerCarIdx, cars[followerCarIdx].TargetPosition);
            }
            leaderCarIdx = followerCarIdx;
            followerCarIdx = cars[leaderCarIdx].followerCarIdx;
        }
    // } else {/*DEBUG*/printf("ERROR: No leader car found @Lane%d\n", laneIdx);
    }
}

__global__ 
void updateActualPositionCUDA(Car* cars) {
    int thrIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
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
    auto microsecs_determineTargetPosition = std::chrono::microseconds::zero();
    auto microsecs_tryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_resolveCollisionsPerLane = std::chrono::microseconds::zero();
    auto microsecs_updateActualPosition = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Memory allocation
    Car* carsDevice;
    checkError(cudaMalloc(&carsDevice, NUM_CARS*sizeof(*carsDevice)));
    int* carIdxDevice /*designed as scalar, implemented as an array amid cudaMalloc*/;
    checkError(cudaMalloc(&carIdxDevice, sizeof(*carIdxDevice)));
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
        // printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        determineTargetPositionCUDA<<<1, NUM_THREADS>>>(carsDevice);
        microsecs_determineTargetPosition += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        start_clock = std::chrono::high_resolution_clock::now();
        tryLaneChangeCUDA<<<1, 1>>>(carsDevice, countLaneChangeDevice) /*No threading*/;
        // tryLaneChangeCUDA<<<1/*single-block*/, NUM_THREADS>>>(carsDevice, countLaneChangeDevice) /*Thread outer loop*/;
        // tryLaneChangeCUDA<<<NUM_BLOCKS/*multi-blocks*/, NUM_THREADS>>>(carsDevice, countLaneChangeDevice) /*Thread outer loop*/;
        // for (int carIdx = 0; carIdx < NUM_CARS; carIdx++) /*Alternative: Thread inner loop*/ {
        //     checkError(cudaMemcpy(carIdxDevice, &carIdx, sizeof(*carIdxDevice), cudaMemcpyHostToDevice));
        //     eachCarTryLaneChangeCUDA<<<1, NUM_THREADS>>>(carsDevice, carIdxDevice, countLaneChangeDevice);
        // }
        microsecs_tryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        resolveCollisionsPerLaneCUDA<<<1, 2>>>(carsDevice);
        microsecs_resolveCollisionsPerLane += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        start_clock = std::chrono::high_resolution_clock::now();
        updateActualPositionCUDA<<<1, NUM_THREADS>>>(carsDevice);
        microsecs_updateActualPosition += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        checkError(cudaMemcpy(cars, carsDevice, NUM_CARS*sizeof(*carsDevice), cudaMemcpyDeviceToHost));
        printStep(fid);
    }
    checkError(cudaMemcpy(&COUNT_LANE_CHANGE, countLaneChangeDevice, sizeof(*countLaneChangeDevice), cudaMemcpyDeviceToHost));
    printf("#Steps: %d, #Lanes: %d, #Cars: %d, #LaneChanges: %d\n", NUM_STEPS, 2, NUM_CARS, COUNT_LANE_CHANGE);
    printf("Total runtime of  determineTargetPosition = %ld us\n", microsecs_determineTargetPosition.count());
    printf("Total runtime of            tryLaneChange = %ld us\n", microsecs_tryLaneChange.count());
    printf("Total runtime of resolveCollisionsPerLane = %ld us\n", microsecs_resolveCollisionsPerLane.count());
    printf("Total runtime of     updateActualPosition = %ld us\n", microsecs_updateActualPosition.count());
    printf("Total runtime of                ALL TASKS = %ld us\n",
            microsecs_determineTargetPosition.count() + microsecs_tryLaneChange.count() +
            microsecs_resolveCollisionsPerLane.count() + microsecs_updateActualPosition.count());

    checkError(cudaFree(carsDevice));
    checkError(cudaFree(carIdxDevice));
    checkError(cudaFree(countLaneChangeDevice));

    free(cars);

    return 0;
}