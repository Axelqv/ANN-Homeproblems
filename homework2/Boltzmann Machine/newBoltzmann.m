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
nrOfHiddenNeurons = length(MValues);
nrOfPatterns = length(allInputs);
trials = 1000;
miniBatches = 20;
k = 2000;
eta = 0.002;
nOut = 3000;
nIn = 2000;
dKLSum = zeros(1,nrOfHiddenNeurons);
counter = 1;



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
plot(MValues, dKL,'DisplayName','Upper Bound')
hold on



% Running the algorithm for the different M-values
for iHiddenNeuron = 1:nrOfHiddenNeurons
    M = MValues(iHiddenNeuron)
    dKLSumCounter = zeros(1,counter);
    for iCounter = 1:counter
    
        
        % Initializing neurons
        visibleNeurons = zeros(N,1);
        hiddenNeurons = zeros(M,1);
        
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
                randomPatternIndex = randi(nrOfPatterns/2);
                feedPattern = XORInputs(:,randomPatternIndex);
                % Initiaize visible neurons as the feed pattern
                visbleNeuron0 = feedPattern;
                
        
                % Update hidden neurons, 
                LocalFieldHidden0 = weights * visbleNeuron0 - thetaHidden;        % Local field, hidden neurons
                for i = 1:M
                    pB0 = probability(LocalFieldHidden0(i));
                    randomNr = rand;
                    if randomNr < pB0
                        hiddenNeurons(i) = 1;
                    else
                        hiddenNeurons(i) = -1;
                    end
                end
                
                
                for t = 1:k
                    % Update visible neurons
                    LocalFieldVisible = weights' * hiddenNeurons - thetaVisible;
                    for j = 1:N
                        pBV = probability(LocalFieldVisible(j));
                        randomNr = rand;
                        if randomNr < pBV
                            visibleNeurons(j) = 1;
                        else
                            visibleNeurons(j) = -1;
                        end
                    end
        
                    % Update hidden neurons
                    LocalFieldHidden = weights * visibleNeurons - thetaHidden;
                    for i = 1:M
                        pBH = probability(LocalFieldHidden(i));
                        randomNr = rand;
                        if randomNr < pBH
                            hiddenNeurons(i) = 1;
                        else
                            hiddenNeurons(i) = -1;
                        end
                    end
                end
        
                % Compute weight and threshold increments
                deltaWeights = deltaWeights + eta*(tanh(LocalFieldHidden0) * visbleNeuron0' - tanh(LocalFieldHidden)*visibleNeurons');
                deltaThetaVisible = deltaThetaVisible - eta*(visbleNeuron0 - visibleNeurons);
                deltaThetaHidden = deltaThetaHidden - eta*(tanh(LocalFieldHidden0) - tanh(LocalFieldHidden));
        
              
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
            visibleNeurons = feedPattern;
            
            LocalFieldHidden =  weights * visibleNeurons - thetaHidden;
            for i = 1:M
                pBH = probability(LocalFieldHidden(i));
                randomNr = rand;
                if randomNr < pBH
                    hiddenNeurons(i) = 1;
                else
                    hiddenNeurons(i) = -1;
                end
            end
        
        
            for iInner = 1:nIn
                LocalFieldVisible = weights' * hiddenNeurons - thetaVisible;
                for j = 1:N
                    pBV = probability(LocalFieldVisible(j));
                    randomNr = rand;
                    if randomNr < pBV
                        visibleNeurons(j) = 1;
                    else
                        visibleNeurons(j) = -1;
                    end
                end
        
                % Update hidden neurons
                LocalFieldHidden = weights * visibleNeurons - thetaHidden;
                for i = 1:M
                    pBH = probability(LocalFieldHidden(i));
                    randomNr = rand;
                    if randomNr < pBH
                        hiddenNeurons(i) = 1;
                    else
                        hiddenNeurons(i) = -1;
                    end
                end
                for iPattern = 1:nrOfPatterns
                    if visibleNeurons == allInputs(:,iPattern)
                        pB(iPattern) = pB(iPattern) + 1/(nIn * nOut);
                    end
                end
            end
        end
        
        
        
        
        % Calculating the dkl
        for mu = 1:8
            if (pData(mu)~=0) && (pB(mu)~=0)  
                dKLSumCounter(iCounter) = dKLSumCounter(iCounter) + pData(mu) * (log(pData(mu)) - log(pB(mu)));
            end
        end

    end
    dKLSum(iHiddenNeuron) = sum(dKLSumCounter)/1;    % Calculating the average out of 3 runs
end

% Plotting the kullback-leiber divergence for the different M-values        
plot(MValues,dKLSum,'ro','DisplayName','D_{KL}')
xlabel('Number of hidden neurons [M]')
ylabel('Kullback-leiber divergence [D_{KL}]')

















            








