function [learningRate, width] = Decay(learningRate, width, learningDecay,...
    widthDecay, epoch)

    learningRate = learningRate * exp(-learningDecay * epoch);
    width = width * exp(-widthDecay * epoch);

end


