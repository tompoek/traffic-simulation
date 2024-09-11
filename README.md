# Traffic Simulation

Simulation of one-way traffic aiming to parallelize when scaling up traffic size (#lanes, #cars)

Compile and run locally:

> ./test.sh

Compile and run in UQ slurm:

> sbatch sbatch_test.sh

Visualize:

> Run visualizeTrafficCars.m in Matlab

Change parameters in the codes (utils.h):

```
const int LANE_LENGTH = 50;
const int NUM_LANES = 4;
constexpr int RANDOM_SEED = 47; // = 0 for fixed scenario
constexpr int NUM_CARS = (RANDOM_SEED > 0) ? 40 : (6 * NUM_LANES); // specify #cars to randomly distribute, or use fixed scenario
const int SAFE_DISTANCE = 2;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100000;
const int TEST_VERSION = 3; // implemented: V2, V3
```

Change parameters in visualization (visualizeTrafficCars.m):

```
LANE_LENGTH = 50; % posIdx = 0:49
NUM_LANES = 4; % laneIdx = 0:3
PAUSE_TIME = 0.2; % The longer pause, the slower
```

Compile and run when profiling / benchmarking performance: (disable printSteps, no visualization)

> Comment out all lines of codes in the main function (SimulateTraffic_cars.cpp) that have comments below:

```
// comment out when profiling
```



## Interim update

__This repo is backed up as a multi-lanes implementation, another work on two-lanes implementation is ongoing. Final decision is TBD!__
