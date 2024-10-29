# Traffic Simulation Multi-Lanes

Simulation of one-way traffic with flexible #lanes aiming to parallelize when scaling up traffic size (both #cars and #lanes). The project focuses on optimizing computing performance i.e. minimizing runtime, rather than building a complex model with fancy visualization. See presentation video:

https://youtu.be/bqS8Da7sjhc

Compile and run locally:

> ./test.sh

Compile and run in UQ slurm:

> sbatch sbatch_test.sh

Visualize:

> Run visualizeTrafficMultiLanes.m in Matlab

Change parameters in the codes (utils.h): 

```
const int NUM_LANES = 4;
constexpr int RANDOM_SEED = 47; // = 0 for fixed scenario
constexpr int NUM_CARS = (RANDOM_SEED > 0) ? 1000 : (6 * NUM_LANES); // specify #cars to randomly distribute, or use fixed scenario
const int LANE_LENGTH = 1000;
const int NUM_THREADS = 256; // Single-block implementation
const int SAFE_DISTANCE = 2;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100;

const int TEST_VERSION = 2; // implemented: V2, V3
```

Change parameters in visualization (visualizeTrafficMultiLanes.m):

```
LANE_LENGTH = 1000; % value must match utils.h, posIdx = 0:LANE_LENGTH-1
NUM_LANES = 4; % value must match utils.h, laneIdx = 0:NUM_LANES-1

PAUSE_TIME = 0.1; % The longer pause, the slower
```

Profiling: 

* Manual timers by default: Results printed on terminal (for local test) or xxx.stdout (for sbatch test)

* Optional: Profile using valgrind, disable printSteps or any visualization related codes (comment out all codes in SimulateTrafficMultiLanes.cpp that call printSteps), then copy paste the following line from test.sh to terminal and run it:

> valgrind --tool=cachegrind ./SimulateTrafficMultiLanes TrafficMultiLanes.csv

## Deprecation of Multi-Lanes version

This multi-lanes implementation has become legacy and is deprecated. Future works may adopt two-lanes version into multi-lanes, instead of reusing this legacy version.
