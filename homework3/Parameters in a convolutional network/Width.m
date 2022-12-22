function width = Width(inputWidth, kernelWidth, paddingLeft, paddingRigth, strideWidth);

    nominator = inputWidth - kernelWidth + paddingLeft + paddingRigth;

    denominator = strideWidth;

    width = nominator / denominator + 1;
    
end

