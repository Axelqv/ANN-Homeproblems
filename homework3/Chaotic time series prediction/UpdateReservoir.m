function newReservoirs = UpdateReservoir(weights, reservoirs, inputWeights, inputNeurons)

    newReservoirs = tanh(weights * reservoirs + inputWeights * inputNeurons);

end

