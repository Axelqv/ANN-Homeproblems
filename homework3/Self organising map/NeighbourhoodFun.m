function funValue = NeighbourhoodFun(pos, minPos, width)

    nominator = norm(pos - minPos)^2;
    denominator = 2 * width^2;
    funValue = exp(-nominator / denominator);

end

