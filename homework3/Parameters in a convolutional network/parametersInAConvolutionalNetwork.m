
clear all
close all
clc

% Conv dimensions
x2 = Width(25, 3, 1, 1, 2)
y2 = Height(25, 3, 1, 1, 2)
z2 = 14;

% Max pooling dimensions
x3 = Width(x2, 2, 1, 1, 1);
y3 = Height(y2, 2, 0, 0, 1);
z3 = 14;


% Conv parameters
convParameters = (3 * 3 * 3 + 1) * 14;


% fully connected layer
nOutputNeurons = 15;
nInputNeurons = x3 * y3 * z3;
fullyConnectedParameters = (nInputNeurons + 1) * nOutputNeurons;

% Output Layer
nInputNeurons = 15;
nOutputNeurons = 10;
outputParameters = (nInputNeurons + 1) * nOutputNeurons;










