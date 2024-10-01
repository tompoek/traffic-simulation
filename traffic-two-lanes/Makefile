
# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -pg -O2
NVFLAGS = -g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O2

# Source files
SRCS = SimulateTrafficTwoLanes.cpp utils.cpp
CUDA_SRCS = SimulateTrafficTwoLanes_CUDA.cu SimulateTrafficTwoLanes_thrust.cu

# Object files
OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

# Output results files
RESULTS = TrafficTwoLanes.csv TrafficTwoLanes_CUDA.csv TrafficTwoLanes_thrust.csv

# Output profile
PROF = gmon.out profile.txt

# Executable name
EXEC = SimulateTrafficTwoLanes SimulateTrafficTwoLanes_CUDA SimulateTrafficTwoLanes_thrust

# Default target
all: $(EXEC)

# Link object files to create the executable
SimulateTrafficTwoLanes: SimulateTrafficTwoLanes.o utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@
SimulateTrafficTwoLanes_CUDA: SimulateTrafficTwoLanes_CUDA.o utils.o
	$(NVCC) $(NVFLAGS) $^ -o $@
SimulateTrafficTwoLanes_thrust: SimulateTrafficTwoLanes_thrust.o utils.o
	$(NVCC) $(NVFLAGS) $^ -o $@

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(EXEC) $(PROF)

.PHONY: all clean
