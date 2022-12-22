function updateWeight = UpdateWinningNeuronWeight(distance, eta, input, weights)

    minDistance = min(min(distance));
    [x, y] = find(distance == minDistance);
    minPos = [x, y];
    neighbourhoodfun = NeighbourhoodFun(minPos, minPos, sigma);
    DeltaWeights(eta, neighbourhoodfun, input, weights);
    updateWeight = weights + DeltaWeights;

end

