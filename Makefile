
# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -pg -O2
NVFLAGS = -g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O2

# Source files
SRCS = SimulateTrafficMultiLanes.cpp utils.cpp
CUDA_SRCS = SimulateTrafficMultiLanes_CUDA.cu

# Object files
OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

# Output results files
RESULTS = TrafficMultiLanes.csv SimulateTrafficMultiLanes_CUDA.csv

# Output profile
PROF = gmon.out profile.txt

# Executable name
EXEC = SimulateTrafficMultiLanes SimulateTrafficMultiLanes_CUDA

# Default target
all: $(EXEC)

# Link object files to create the executable
SimulateTrafficMultiLanes: SimulateTrafficMultiLanes.o utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@
SimulateTrafficMultiLanes_CUDA: SimulateTrafficMultiLanes_CUDA.o utils.o
	$(NVCC) $(NVFLAGS) $^ -o $@

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(EXEC) $(RESULTS) $(PROF)

.PHONY: all clean
