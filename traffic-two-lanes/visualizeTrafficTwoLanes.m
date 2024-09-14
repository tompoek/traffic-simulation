%% 
close all
clear all

data = readmatrix('TrafficTwoLanes.csv');
[NUM_STEPS, NUM_CARS] = size(data);
% NUM_CARS = NUM_CARS / 2; % Each car has two values: laneIdx and posIdx
NUM_CARS = 20; % Too many cars now, only track the first 20
LANE_LENGTH = 2000; % posIdx = 0:2047
NUM_LANES = 2; % laneIdx = 0:1

%% 
% Plot horizontal lanes (laneIdx = 0 at the top, 3 at the bottom).
% Plot car positions in lane (posIdx = 0 at the leftmost, 49 at rightmost)
figure;

CAR_SIZE = 0.6; % Size of cars to plot
PAUSE_TIME = 0.0005; % The longer pause, the slower

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
                    posIdx-20, NUM_LANES-laneIdx-0.05; 
                    posIdx-20, NUM_LANES-laneIdx+0.05];
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