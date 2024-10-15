#include <iostream>
#include <chrono>
#include <curand_kernel.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/find.h>

#include "utils.h"

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
    __host__ __device__ void operator()(Car &car) {
        car.TargetPosition = car.Position + car.TargetSpeed;
    }
};

struct UpdateActualPosition {
    __host__ __device__ void operator()(Car &car) {
        car.Position = car.TargetPosition;
        // car.TargetSpeed = /*randomly change target speed to increase traffic dynamics.*/ (curand(&state)%SPEED_LIMIT) + 1;
    }
};

struct IsThisLanesFirstLeader {
    int laneIdx;
    __host__ __device__ int operator()(const Car& car) const {
        return int(car.leaderCarIdx < 0 && car.laneIdx == laneIdx);
    }
};

struct IsAtCurrentLane {
    int laneIdx;
    __host__ __device__ int operator()(const Car& car) const {
        return int(car.laneIdx == laneIdx);
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
    free(numCarsInLanes); // only for init
    free(carIndicesInLanes); // only for init
    printStepThrustHost(fid, carsHost); // comment out when profiling

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        printf("@ Step %d\n", step);

        // ALL CARS TRY LANE CHANGE
        start_clock = std::chrono::high_resolution_clock::now();
        //TODO
        thrust::for_each(carsDevice.begin(), carsDevice.end(), DetermineTargetPosition());
        microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        // ALL CARS DRIVE FORWARD
        start_clock = std::chrono::high_resolution_clock::now();
        //TODO
        
        // resolve collisions if any
        // <<<<<< SEGMENTATION FAULT >>>>>>
        // for (int laneIdx = 0; laneIdx < 2; laneIdx++) {
        //     int leaderCarIdx = thrust::distance(carsDevice.begin(), thrust::find_if(carsDevice.begin(), carsDevice.end(), IsThisLanesFirstLeader{laneIdx}));
        //     if (leaderCarIdx < NUM_CARS /*if no leader car found, thrust::find_if would return NUM_CARS*/) {
        //         Car* carsDeviceRawPtr = thrust::raw_pointer_cast(carsDevice.data());
        //         int followerCarIdx = carsDeviceRawPtr[leaderCarIdx].followerCarIdx;
        //         thrust::device_vector<int> iAmAtCurrentLane(NUM_CARS);
        //         thrust::transform(carsDevice.begin(), carsDevice.end(), iAmAtCurrentLane.begin(), IsAtCurrentLane{laneIdx});
        //         int numCarsAtCurrentLane = thrust::reduce(iAmAtCurrentLane.begin(), iAmAtCurrentLane.end(), 0, thrust::plus<int>());
        //         //TODO: Fix segmentation fault in for loop below.
        //         // for (int laneCarIdx = 0; laneCarIdx < numCarsAtCurrentLane; laneCarIdx++) {
        //         //     if (followerCarIdx == -1) break;
        //         //     if (carsDeviceRawPtr[followerCarIdx].TargetPosition >= carsDeviceRawPtr[leaderCarIdx].TargetPosition /*my follower would hit me*/) {
        //         //         carsDeviceRawPtr[followerCarIdx].TargetPosition = carsDeviceRawPtr[leaderCarIdx].TargetPosition - 1/*tell them to move a distance behind me*/;
        //         //     }
        //         //     leaderCarIdx = followerCarIdx;
        //         //     followerCarIdx = carsDeviceRawPtr[leaderCarIdx].followerCarIdx;
        //         // }
        //     } else {
        //         printf("ERROR: No leader car found @Lane%d\n", laneIdx);
        //         return;
        //     }
        // }
        // <<<<<< SEGMENTATION FAULT >>>>>>
        resolveCollisionsThreadLanesCUDA<<<1, 2>>>(thrust::raw_pointer_cast(carsDevice.data()));

        // update actual position
        thrust::for_each(carsDevice.begin(), carsDevice.end(), UpdateActualPosition());
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);


        carsHost = carsDevice; // comment out when profiling
        printStepThrustHost(fid, carsHost); // comment out when profiling
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());


    free(cars);


    return 0;
}