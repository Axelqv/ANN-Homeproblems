% Self organising map. Written by: Axel Qvarnström
clear all 
close all
clc

% Loading the data and labels
data = load('iris-data.csv');
labels = load('iris-labels.csv');

% Standardise the data
data = data ./ max(data);

% Initializing
weightArray = rand([40,40,4]);
learningRate = 0.1;
learningDecay = 0.01;
width = 10;
sigmaDecay = 0.05;


nEpochs = 10;
nDataPoints = length(data);




for i = 1:150
    randomDataPointIndex = randi(nDataPoints);
    sumUnderSqrt = 0;
    input = data(randomDataPointIndex,:);
    term1 = (weightArray(:,:,1) - input(1)).^2
    term2 = (weightArray(:,:,2) - input(2)).^2
    term3 = (weightArray(:,:,3) - input(3)).^2
    term4 = (weightArray(:,:,4) - input(4)).^2
    distance = sqrt(term1 + term2 + term3 + term4);
%     for j = 1:size(data,2)
%         jInput = input(j);
%         weights = squeeze(weightArray(:,:,j));
%         sumUnderSqrt = sumUnderSqrt + (weights - jInput).^2;      % Calculating the summation for the distance, the only thing left is to take the square root of it
%     end
%     distance = sqrt(sumUnderSqrt);

    % updating Weights for the winning neuron
    minDistance = min(distance,[],'all');
    [x,y] = find(distance == minDistance);
    minPos = [x, y];
    weights2Update = squeeze(weightArray(x,y,:))';
    neighbourhoodfun = NeighbourhoodFun(minPos, minPos, width);
    deltaWeights = DeltaWeights(learningRate, neighbourhoodfun, input, weights2Update);
    weights2Update = weights2Update + deltaWeights;
    weightArray(x,y,:) = weights2Update;

    % Updating Weights for the close neurons
    weightArray = UpdateWeightsForCloseNeurons(distance, learningRate, input, weightArray, minPos,width);

end











    




