function deltaW = DeltaWeights(learningRate, neighbourhoodFun, inputs, Weight2Update)
   
    deltaW = learningRate * neighbourhoodFun * (inputs - Weight2Update);

end

