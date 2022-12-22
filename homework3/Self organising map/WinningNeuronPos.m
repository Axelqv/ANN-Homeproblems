function winningNeuronPos = WinningNeuronPos(distance)

    minDistance = min(distance, [],"all");
    [x,y] = find(distance == minDistance);
    winningNeuronPos = [x, y];

end

