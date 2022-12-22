function weightArray = UpdateWeightsForCloseNeurons(distance, learningRate, input,...
    weightArray, minPos,width)

    closeDistances = distance(distance < 3*width);
    nDistances = length(closeDistances);
    

    for i = 1:nDistances
        [x, y] = find(distance == closeDistances(i));
        closePos = [x, y];
        weights2Update = squeeze(weightArray(x,y,:))';
        neighbourhoodfun = NeighbourhoodFun(closePos, minPos, width);
        deltaWeights = DeltaWeights(learningRate, neighbourhoodfun, input, weights2Update);
        weights2Update = weights2Update + deltaWeights;
        weightArray(x,y,:) = weights2Update;
    end
    
end

