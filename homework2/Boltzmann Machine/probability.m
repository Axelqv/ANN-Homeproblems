function prob = probability(bHidden)

    denominator = 1 + exp(-2*bHidden);
    prob = 1 / denominator;

end

