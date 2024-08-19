%% 
close all
clear all

data = readmatrix('trafficCars.csv');
[NUM_STEPS, NUM_CARS] = size(data);
NUM_CARS = NUM_CARS / 2; % Each car has two values: laneIdx and posIdx
LANE_LENGTH = 50; % posIdx = 0:49
NUM_LANES = 4; % laneIdx = 0:3

%% 
% Plot horizontal lanes (laneIdx = 0 at the top, 3 at the bottom).
% Plot car positions in lane (posIdx = 0 at the leftmost, 49 at rightmost)
figure;

CAR_SIZE = 0.6; % Size of cars to plot
PAUSE_TIME = 0.2; % The longer pause, the slower

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
                    posIdx-CAR_SIZE, NUM_LANES-laneIdx-CAR_SIZE/8; 
                    posIdx-CAR_SIZE, NUM_LANES-laneIdx+CAR_SIZE/8];
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