
# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Wall -pg -O2
NVFLAGS = -g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O2

# Source files
SRCS = SimulateTraffic_cars.cpp utils.cpp
CUDA_SRCS = SimulateTraffic_cuda.cu

# Object files
OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)

# Output results files
RESULTS = trafficCars.csv trafficCars_cuda.csv

# Output profile
PROF = gmon.out profile.txt

# Executable name
EXEC = SimulateTraffic_cars SimulateTraffic_cuda

# Default target
all: $(EXEC)

# Link object files to create the executable
SimulateTraffic_cars: SimulateTraffic_cars.o utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@
SimulateTraffic_cuda: SimulateTraffic_cuda.o utils.o
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
