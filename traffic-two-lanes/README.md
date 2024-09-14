# Traffic Two Lanes

Key differences to Multi-Lanes implementation:

* Simplified to two lanes only, making the traffic denser, motivating cars to change lane.

* No more lane wrapping. Lane wrapping uses LANE_LENGTH, which only affects initialization and visualization.

* At initialization, cars of smaller indices are placed ahead of those of larger indices (only affects CPU loop sequence).

* Introduce randomness, change target speed for all cars at the end of each step.

* GPU parallelization.
