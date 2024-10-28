
# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -pg -O2

# Source files
SRCS = SimulateTrafficMultiLanes.cpp utils.cpp

# Object files
OBJS = $(SRCS:.cpp=.o) 

# Output results files
RESULTS = TrafficMultiLanes.csv 

# Output profile
PROF = gmon.out profile.txt

# Executable name
EXEC = SimulateTrafficMultiLanes 

# Default target
all: $(EXEC)

# Link object files to create the executable
SimulateTrafficMultiLanes: SimulateTrafficMultiLanes.o utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(EXEC) $(RESULTS) $(PROF)

.PHONY: all clean
