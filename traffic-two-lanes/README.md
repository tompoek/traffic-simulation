# Traffic Two Lanes

Simulation of one-way two-lane traffic aiming to parallelize when scaling up traffic size (#cars)

Compile and run locally:

> ./test.sh

Compile and run in UQ slurm:

> sbatch sbatch_test.sh

Visualize:

> Run visualizeTrafficTwoLanes.m in Matlab

Change parameters in the codes (utils.h):

```
const int RANDOM_SEED = 47;
const int NUM_CARS = 500;
const int LANE_LENGTH = 1.5 * NUM_CARS;
const int NUM_THREADS = 4; // must be exponential of 2
const int NUM_BLOCKS = (NUM_CARS + NUM_THREADS - 1) / NUM_THREADS;
const int SPEED_LIMIT = 4;
const int NUM_STEPS = 100;
```

Change parameters in visualization (visualizeTrafficTwoLanes.m):

```
PAUSE_TIME = 0.1; % The longer pause, the slower
```

Compile and run when profiling / benchmarking performance:

> Optional for faster profiling: Disable printSteps / visualizations in the main function (SimulateTrafficTwoLanes.cpp) that have comments below:

```
// comment out when profiling
```

## Motivation of Two-Lanes vs Multi-Lanes implementation:

* Fair benchmarking <- Equal chances of lane change <- Constant traffic density <- Increase lane length proportionally to increase of #cars

* Model gets simpler instructions, optimized for GPU parallelization.

## Key differences of Two-Lanes to Multi-Lanes implementation:

* Simplified to two lanes only, making the traffic denser, motivating cars to change lane.

* No more lane wrapping. Lane wrapping uses LANE_LENGTH, which only affects initialization and visualization.

* At initialization, cars of smaller indices are placed ahead of those of larger indices (only affects CPU loop sequence).

* GPU parallelization implemented.
