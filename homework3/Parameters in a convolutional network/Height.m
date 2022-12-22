function height = Height(inputHeight, kernelHeight, paddingTop, paddingBottom, strideHeight)

    nominator = inputHeight - kernelHeight + paddingTop + paddingBottom;
    
    denominator = strideHeight;

    height = nominator / denominator + 1;

end


