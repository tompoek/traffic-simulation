#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"

int COUNT_LANE_CHANGE = 0; // for profiling number of successful lane changes

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

bool tryLaneChangeCUDA(CarV3* &cars, LaneV3 &lane, LaneV3 &targetLane, int &laneCarIdx, int &laneIdx, int &targetLaneIdx) {
    bool carHasChangedLane = false;
    bool targetLaneIsSafe = true;
    for (int ii=0; ii<targetLane.numCars; ++ii) {
        int distToCarii = ((cars[targetLane.CarIndices[ii]].Position - cars[lane.CarIndices[laneCarIdx]].Position) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
        if (distToCarii < SAFE_DISTANCE) {
            targetLaneIsSafe = false;
            break;
        }
    }
    if (targetLaneIsSafe) {
        // printf("Car %d changes from Lane %d to %d\n", lane.CarIndices[laneCarIdx], laneIdx, targetLaneIdx);
        COUNT_LANE_CHANGE++;
        // EXECUTE LANE CHANGE <<< START
        int &carIdx = lane.CarIndices[laneCarIdx];
        CarV3 &carToMove = cars[carIdx]; // alias the car to move
        // find which index to insert the car, based on position
        int toLaneInsertIndex = 0;
        for (int i=targetLane.numCars-1; i>=0; --i) {
            if (cars[targetLane.CarIndices[i]].Position < carToMove.Position) { // == case should not happen, because lane must be safe!
                toLaneInsertIndex = i;
                break;
            }
        }
        // shift all car indices in target lane after toLaneInsertIndex
        for (int i = targetLane.numCars; i > toLaneInsertIndex; --i) {
            targetLane.CarIndices[i] = targetLane.CarIndices[i - 1]; // THIS IS THE EXPECTED OPTMIZATION OF V3 vs V2
        }
        // insert the car index to the index found
        targetLane.CarIndices[toLaneInsertIndex] = carIdx; // THIS IS THE EXPECTED OPTMIZATION OF V3 vs V2
        // assuming no lead car, drive at target speed, TBD: introduce acceleration model
        carToMove.Speed = carToMove.TargetSpeed;
        targetLane.numCars++;
        // shift the moved car's index to the rightmost side
        for (int i=laneCarIdx; i<lane.numCars; ++i) {
            lane.CarIndices[i] = lane.CarIndices[i+1];
        }
        // delete the moved car from previous lane
        lane.CarIndices[lane.numCars - 1] = 0;
        lane.numCars--;
        // EXECUTE LANE CHANGE <<< FINISH
        carHasChangedLane = true;
    }
    return carHasChangedLane;
}

void allCarsTryLaneChangeCUDA(CarV3* &cars, LaneV3* lanes, int &laneIdx, int* &numCarsInLanes, int* &carIndicesInLanes) {
    LaneV3 &lane = lanes[laneIdx];
    bool hasLeadCar = false;
    bool hasNextLane = true;
    int nextLaneIdx = laneIdx + 1;
    if (laneIdx == NUM_LANES - 1) {hasNextLane = false; nextLaneIdx = 0;}
    LaneV3 &nextLane = lanes[nextLaneIdx];
    bool hasPreviousLane = true;
    int previousLaneIdx = laneIdx - 1;
    if (laneIdx == 0) {hasPreviousLane = false; previousLaneIdx = NUM_LANES - 1;}
    LaneV3 &previousLane = lanes[previousLaneIdx];
    int &numCarsInCurrentLane = numCarsInLanes[laneIdx];
    if (numCarsInCurrentLane > 1) {
        for (int i = numCarsInCurrentLane-2; i >= 0; --i) {
            hasLeadCar = false;
            int &carIdxI = carIndicesInLanes[laneIdx*LANE_LENGTH + i]; // lane.CarIndices[i];
            int &carPos = cars[carIdxI].Position;
            for (int j = i+1; j < numCarsInCurrentLane; ++j) {
                int &carIdxJ = carIndicesInLanes[laneIdx*LANE_LENGTH + j]; // lane.CarIndices[j];
                // detect lead car in the current lane
                int distToCarj = ((cars[carIdxJ].Position - carPos) % LANE_LENGTH + LANE_LENGTH) % LANE_LENGTH;
                if (distToCarj < SAFE_DISTANCE) {
                    hasLeadCar = true;
                    break;
                }
            }
            if (hasLeadCar) {
                bool carHasChangedLane = false;
                // detect cars in the target lane
                if (hasNextLane) {
                    carHasChangedLane = tryLaneChangeCUDA(cars, lane, nextLane, i, laneIdx, nextLaneIdx);
                }
                if (!carHasChangedLane && hasPreviousLane) {
                    carHasChangedLane = tryLaneChangeCUDA(cars, lane, previousLane, i, laneIdx, previousLaneIdx);
                }
            }
        }
    }
}

__global__ 
void allCarsDriveForwardCUDA(CarV3* carsDevice, int* numCarsInLanesDevice, int* carIndicesInLanesDevice) {
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // Determine target positions
    for (int i = thrIdx; i < NUM_CARS; i += stride) {
        carsDevice[i].TargetPosition = carsDevice[i].Position + carsDevice[i].Speed;
    }
    // All Lanes Resolve Collision Conflict
    for (int laneIdx = thrIdx; laneIdx < NUM_LANES; laneIdx += stride) {
        for (int i = numCarsInLanesDevice[laneIdx] - 1; i > 0; i--) {
            int num_collisions = 0;
            // int &carIdxI = lane.CarIndices[i];
            int &carIdxI = carIndicesInLanesDevice[laneIdx*LANE_LENGTH + i];
            for (int j = i - 1; j >= 0; j--) {
                // int &carIdxJ = lane.CarIndices[j];
                int &carIdxJ = carIndicesInLanesDevice[laneIdx*LANE_LENGTH + j];
                if (carsDevice[carIdxJ].TargetPosition >= carsDevice[carIdxI].TargetPosition) { // ASSUMPTION: speeds never exceeds LANE_LENGTH
                    // Collision detected, move car j as close as possible without colliding
                    num_collisions++;
                    carsDevice[carIdxJ].TargetPosition = carsDevice[carIdxI].TargetPosition - num_collisions; // ... - num_collisions * SAFE_DISTANCE;
                    carsDevice[carIdxJ].Speed = carsDevice[carIdxI].Speed; // then adjust car j's speed to lead car i's speed
                }
            }
        }
    }
    // Update positions after conflicts resolved
    for (int i = thrIdx; i < NUM_CARS; i += stride) {
        carsDevice[i].Position = ((carsDevice[i].TargetPosition % LANE_LENGTH) + LANE_LENGTH) % LANE_LENGTH;
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
    // LaneV2* lanesV2 = static_cast<LaneV2*>(malloc(sizeof(LaneV2) * NUM_LANES));
    CarV3* carsV3 = static_cast<CarV3*>(malloc(sizeof(*carsV3)* NUM_CARS));
    //<<<TO DELETE>>>
    LaneV3* lanesV3 = static_cast<LaneV3*>(malloc(sizeof(*lanesV3) * NUM_LANES));
    for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
        lanesV3[lane_index].numCars = 0;
        lanesV3[lane_index].CarIndices = static_cast<int*>(malloc(sizeof(int) * LANE_LENGTH));
    }
    //<<<TO DELETE>>>

    int* numCarsInLanes = static_cast<int*>(malloc(sizeof(int) * NUM_LANES));
    int* carIndicesInLanes = static_cast<int*>(malloc(sizeof(int) * NUM_LANES * LANE_LENGTH));

    CarV3* carsDevice;
    checkError(cudaMalloc(&carsDevice, NUM_CARS*sizeof(*carsDevice)));
    int* numCarsInLanesDevice;
    checkError(cudaMalloc(&numCarsInLanesDevice, NUM_LANES*sizeof(int)));
    int* carIndicesInLanesDevice;
    checkError(cudaMalloc(&carIndicesInLanesDevice, NUM_LANES*LANE_LENGTH*sizeof(int)));

    // Initialization
    initializeTrafficCUDA(carsV3, numCarsInLanes, carIndicesInLanes);

    printStepCarsCUDA(fid, carsV3, numCarsInLanes, carIndicesInLanes); // comment out when profiling

    // Simulation loop
    for (int step=0; step<NUM_STEPS; ++step) {
        
            //<<<JUST DEBUGGING >>>
        // Intermediate: Convert numCarsInLanes and carIndicesIndicesInLanes to lanesV3
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            lanesV3[laneIdx].numCars = numCarsInLanes[laneIdx];
            for (int laneCarIdx = 0; laneCarIdx < numCarsInLanes[laneIdx]; laneCarIdx++) {
                lanesV3[laneIdx].CarIndices[laneCarIdx] = carIndicesInLanes[laneIdx*LANE_LENGTH + laneCarIdx];
            }
        }
            //<<<JUST DEBUGGING >>>

        // printf("@ Step %d\n", step);
        // Try Lane change
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            start_clock = std::chrono::high_resolution_clock::now();
            
            //TODO: GET RID OF lanesV3
            allCarsTryLaneChangeCUDA(carsV3, lanesV3, laneIdx, numCarsInLanes, carIndicesInLanes);

            //<<<JUST DEBUGGING >>>
            // allCarsTryLaneChangeV3(carsV3, lanesV3, laneIdx);
            //<<<JUST DEBUGGING >>>

            microsecs_allCarsTryLaneChange += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);
        }

            //<<<JUST DEBUGGING >>>
        // Intermediate: Convert lanesV3 to numCarsInLanes and carIndicesIndicesInLanes
        for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
            numCarsInLanes[laneIdx] = lanesV3[laneIdx].numCars;
            for (int laneCarIdx = 0; laneCarIdx < numCarsInLanes[laneIdx]; laneCarIdx++) {
                carIndicesInLanes[laneIdx*LANE_LENGTH + laneCarIdx] = lanesV3[laneIdx].CarIndices[laneCarIdx];
            }
        }
            //<<<JUST DEBUGGING >>>

        // All cars drive forward, must resolve collisions before updating positions
        start_clock = std::chrono::high_resolution_clock::now();
        // Copy mem host to device
        checkError(cudaMemcpy(carsDevice, carsV3, NUM_CARS*sizeof(*carsDevice), cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(numCarsInLanesDevice, numCarsInLanes, NUM_LANES*sizeof(*numCarsInLanes), cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(carIndicesInLanesDevice, carIndicesInLanes, NUM_LANES*sizeof(*carIndicesInLanes), cudaMemcpyHostToDevice));

        allCarsDriveForwardCUDA<<<1, NUM_THREADS>>>(carsDevice, numCarsInLanesDevice, carIndicesInLanesDevice); // Single-block implementation
        // allCarsDriveForwardCUDA<<<NUM_LANE_BLOCKS, NUM_CAR_THREADS>>>(carsDevice, numCarsInLanesDevice, carIndicesInLanesDevice); // Multi-block implementation

            //<<<JUST DEBUGGING >>>
        // for (int lane_index = 0; lane_index < NUM_LANES; ++lane_index) {
        //     allCarsDriveForwardV3(carsV3, lanesV3, lane_index);
        // }
            //<<<JUST DEBUGGING >>>

        checkError(cudaDeviceSynchronize());
        // Copy mem device to host
        checkError(cudaMemcpy(carsV3, carsDevice, NUM_CARS*sizeof(*carsDevice), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(numCarsInLanes, numCarsInLanesDevice, NUM_LANES*sizeof(*numCarsInLanes), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(carIndicesInLanes, carIndicesInLanesDevice, NUM_LANES*sizeof(*carIndicesInLanes), cudaMemcpyDeviceToHost));
        microsecs_allCarsDriveForward += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clock);

        printStepCarsCUDA(fid, carsV3, numCarsInLanes, carIndicesInLanes); // comment out when profiling
    }
    printf("Num Steps: %d, Num Lanes: %d, Num Cars: %d\n", NUM_STEPS, NUM_LANES, NUM_CARS);
    printf("Num of successful lane changes = %d\n", COUNT_LANE_CHANGE);
    printf("Cumulative microseconds of allCarsTryLaneChange = %ld us\n", microsecs_allCarsTryLaneChange.count());
    printf("Cumulative microseconds of allCarsDriveForward = %ld us\n", microsecs_allCarsDriveForward.count());

    // free(lanesV2);
    free(carsV3);
    free(lanesV3);
    free(numCarsInLanes);
    free(carIndicesInLanes);
    checkError(cudaFree(numCarsInLanesDevice));
    checkError(cudaFree(carIndicesInLanesDevice));
    checkError(cudaFree(carsDevice));

    return 0;
}
