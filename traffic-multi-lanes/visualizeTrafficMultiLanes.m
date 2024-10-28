%% 
close all
clear all

data = readmatrix('TrafficMultiLanes.csv');
[NUM_STEPS, NUM_CARS] = size(data);
NUM_CARS = NUM_CARS / 2; % Each car has two values: laneIdx and posIdx
NUM_CARS = 20; % If too many cars, only track the first few
NUM_STEPS = 100; % If too many steps, only visualize the first few
LANE_LENGTH = 1000; % value must match utils.h, posIdx = 0:LANE_LENGTH-1
NUM_LANES = 4; % value must match utils.h, laneIdx = 0:NUM_LANES-1

%% 
% Plot horizontal lanes (laneIdx = 0 at the top, 3 at the bottom).
% Plot car positions in lane (posIdx = 0 at the leftmost, 49 at rightmost)
figure;

PAUSE_TIME = 0.1; % The longer pause, the slower

% % Define palette for colors with increasing NUM_CARS
colors = hsv(NUM_CARS);

for step = 1:NUM_STEPS % Each step is an image frame.
    clf;
    hold on;
    grid on;
    
    % Each carIdx uses the same unique color
    for carIdx = 0:(NUM_CARS-1)
        laneIdx = data(step, 2*carIdx + 1);
        posIdx = data(step, 2*carIdx + 2);
        carColor = colors(carIdx + 1, :);
        % Show car as right-pointing triangle.
        vertices = [posIdx, NUM_LANES-laneIdx; 
                    posIdx-0.01*LANE_LENGTH, NUM_LANES-laneIdx-0.025*NUM_LANES; 
                    posIdx-0.01*LANE_LENGTH, NUM_LANES-laneIdx+0.025*NUM_LANES];
        fill(vertices(:,1), vertices(:,2), carColor, 'EdgeColor', 'k');
    end
    
    % Set axis limits to fit the lanes and positions
    xlim([-1, LANE_LENGTH + 1]);
    ylim([0.5, NUM_LANES + 0.5]);
    yticks(1:NUM_LANES);
    
    % Update the plot to show the current frame
    drawnow;
    pause(PAUSE_TIME);
end