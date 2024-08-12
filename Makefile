
# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -pg

# Source files
SRCS = SimulateTraffic_cars.cpp utils.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output results files
RESULTS = trafficSpaceOccupancy.csv

# Output profile
PROF = gmon.out profile.txt

# Executable name
EXEC = SimulateTraffic_cars

# Default target
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(EXEC) $(RESULTS) $(PROF)

.PHONY: all clean
