% Boltzmann machine written by Axel Qvarnström
clear all
close all
clc

% Inputs
XORInputs = [-1,-1,-1; 1,-1,1; -1,1,1; 1,1,-1]';
allInputs = [-1,-1,-1; 1,-1,1; -1,1,1; 1,1,-1; 1,1,1; 1,-1,-1; -1,-1,1; -1,1,-1]';

% Probability distribution for the data
pData = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0];

% Parameters
N = 3;
MValues = [1, 2, 4, 8];
nrOfHiddenNeurons = length(MValues)
nrOfPatterns = length(allInputs);
trials = 1000;
miniBatches = 20;
k = 200;
eta = 0.01;
nOut = 3000;
nIn = 2000;
dKLSum = zeros(1,nrOfHiddenNeurons);



% Plotting the boundary D_kl
dKL = zeros(1, length(MValues));
for i = 1:length(MValues)
    M = MValues(i);
    if M < 2^(N-1)-1
    dKL(i) = N - log2(M+1) - ((M+1)/(2^(log2(M + 1))));
    else
    dKL(i) = 0;
    end
end
plot(MValues, dKL)
hold on



% Running the algorithm for the different M-values
for iHiddenNeuron = 1:nrOfHiddenNeurons
    M = MValues(iHiddenNeuron)
    
    
    % Initializing neurons
    v = zeros(N,1);
    h = zeros(M,1);
    
    % Initial weights and thresholds
    weights = normrnd(0, 1, [M,N]);
    for i = 1:size(weights,1) 
        for j = 1:size(weights,2)
            if j == i
                weights(i,i) = 0;      % Making the diagonal weights to zero
            end
        end
    end
    
    % Initialize thresholds
    thetaHidden = zeros(M,1);
    thetaVisible = zeros(N,1);

    for itrial = 1:trials
        % Initialize the errors
        deltaWeights = zeros(M,N);
        deltaThetaHidden = zeros(M,1);
        deltaThetaVisible = zeros(N,1);
    
        for iMiniBatch = 1: miniBatches
            % Pick one pattern randomly from x1-x4
            randPatternIndex = randi(nrOfPatterns/2);
            feedPattern = XORInputs(:,randPatternIndex);
            % Initiaize visible neurons as the feed pattern
            v0 = feedPattern;
            
    
            % Update hidden neurons, 
            b0H = weights * v0 - thetaHidden;        % Local field, hidden neurons
            for i = 1:M
                pB0 = probability(b0H(i));
                randomNr = rand;
                if randomNr < pB0
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            
            
            for t = 1:k
                % Update visible neurons
                bV = weights' * h - thetaVisible;
                for j = 1:N
                    pBV = probability(bV(j));
                    randomNr = rand;
                    if randomNr < pBV
                        v(j) = 1;
                    else
                        v(j) = -1;
                    end
                end
    
                % Update hidden neurons
                bH = weights * v - thetaHidden;
                for i = 1:M
                    pBH = probability(bH(i));
                    randomNr = rand;
                    if randomNr < pBH
                        h(i) = 1;
                    else
                        h(i) = -1;
                    end
                end
            end
    
            % Compute weight and threshold increments
            deltaWeights = deltaWeights + eta*(tanh(b0H) * v0' - tanh(bH)*v');
            deltaThetaVisible = deltaThetaVisible - eta*(v0 - v);
            deltaThetaHidden = deltaThetaHidden - eta*(tanh(b0H) - tanh(bH));
    
          
        end
        % Updating weight and threshold
        weights = weights + deltaWeights;
        thetaVisible = thetaVisible + deltaThetaVisible;
        thetaHidden = thetaHidden + deltaThetaHidden;
    end
    
    pB = zeros(nrOfPatterns,1);
    for iOuter = 1:nOut
        randomPatternIndex = randi(nrOfPatterns);
        feedPattern = allInputs(:,randomPatternIndex);
        % Initiaize visible neurons as the feed pattern
        v = feedPattern;
        
        bH =  weights * v - thetaHidden;
        for i = 1:M
            pBH = probability(bH(i));
            randomNr = rand;
            if randomNr < pBH
                h(i) = 1;
            else
                h(i) = -1;
            end
        end
    
    
        for iInner = 1:nIn
            bV = weights' * h - thetaVisible;
            for j = 1:N
                pBV = probability(bV(j));
                randomNr = rand;
                if randomNr < pBV
                    v(j) = 1;
                else
                    v(j) = -1;
                end
            end
    
            % Update hidden neurons
            bH = weights * v - thetaHidden;
            for i = 1:M
                pBH = probability(bH(i));
                randomNr = rand;
                if randomNr < pBH
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            for iPattern = 1:nrOfPatterns
                if v == allInputs(:,iPattern)
                    pB(iPattern) = pB(iPattern) + 1/(nIn * nOut);
                end
            end
        end
    end
    
    
    
    
    % Calculating the dkl
    for mu = 1:8
        if pData(mu) ~= 0
            dKLSum(iHiddenNeuron) = dKLSum(iHiddenNeuron) + pData(mu) * (log(pData(mu)) - log(pB(mu)));
        end
    end
    
   
end

plot(MValues,dKLSum,'ro')
















            








