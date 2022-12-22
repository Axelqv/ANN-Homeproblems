function outputWeights = RidgeRegression(R, k, xTrain, identityMatrix)

    outputWeights = xTrain * R' *(R * R' + k*identityMatrix);

end



    