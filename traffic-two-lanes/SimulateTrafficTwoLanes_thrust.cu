#include <iostream>
#include <chrono>
#include <curand_kernel.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/find.h>

#include "utils.h"

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

__host__
void printStepThrustHost(FILE* &fid, thrust::host_vector<Car> &carsHost) {
    for (int carIdx = 0; carIdx < NUM_CARS; ++carIdx) {
        if (carIdx>0) fprintf(fid, ",");
        int laneIdx = carsHost[carIdx].laneIdx;
        int position = carsHost[carIdx].Position;
        fprintf(fid, "%d,%d", laneIdx, (position % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH);
    }
    fprintf(fid, "\n");
}

struct DetermineTargetPosition {
    __device__ void operator()(Car &car) {
        car.TargetPosition = car.Position + car.TargetSpeed;
    }
};

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
        }
    }
}

struct UpdateActualPosition {
    __device__ void operator()(Car &car) {
        car.Position = car.TargetPosition;
        // car.TargetSpeed = /*randomly change target speed to increase traffic dynamics.*/ (curand(&state)%SPEED_LIMIT) + 1;
    }
};

__global__
void resolveCollisionsThreadLanesCUDA(Car* cars) {
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
    } else {
        printf("ERROR: No leader car found @Lane%d\n", laneIdx);
        return;
    }
}

int main(int argc, char** argv) {
    thrust::host_vector<Car> carsHost(NUM_CARS);
    thrust::device_vector<Car> carsDevice(NUM_CARS);
    // Measure runtime
    std::chrono::high_resolution_clock::time_point start_clock; // used by all timers
    auto microsecs_allCarsTryLaneChange = std::chrono::microseconds::zero();
    auto microsecs_allCarsDriveForward = std::chrono::microseconds::zero();

    // Prepare for printing to file
    FILE* fid = argc > 1 ? fopen(argv[1], "w") : stdout; // comment out when profiling

    // Initialization
    initializeTrafficTwoLanes();
    std::copy(cars, cars + NUM_CARS, carsHost.begin());
    carsDevice = carsHost;
    thrust::device_vector<int> countLaneChangeDevice(1, 0);
    free(numCarsInLanes); // only for init
    free(carIndicesInLanes); // only for init
    printStepThrustHost(fid, carsHost); // comment out when profiling

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        // printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        thrust::for_each(carsDevice.begin(), carsDevice.end(), DetermineTargetPosition());
        tryLaneChangeCUDA<<<1, NUM_THREADS>>>(thrust::raw_pointer_cast(carsDevice.data()), thrust::raw_pointer_cast(countLaneChangeDevice.data()));
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        resolveCollisionsThreadLanesCUDA<<<1, 2>>>(thrust::raw_pointer_cast(carsDevice.data()));
        thrust::for_each(carsDevice.begin(), carsDevice.end(), UpdateActualPosition());
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        carsHost = carsDevice; // comment out when profiling
        printStepThrustHost(fid, carsHost); // comment out when profiling
    }
    thrust::copy(countLaneChangeDevice.begin(), countLaneChangeDevice.end(), &COUNT_LANE_CHANGE);
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());

    free(cars);


    return 0;
}