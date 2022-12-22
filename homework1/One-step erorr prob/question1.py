import numpy as np
import random
from time import time

start_time = time()
# The nr of patterns
p_vector = [12, 24, 48, 70, 100, 120]
# the number of bits
N = 120
# Creating a errors variable that will grow when errors occur( when updated_bit != bit)
start_time = time()
probability_error = []
for p in p_vector:
    errors = 0
    for _ in range(10 ** 3):

        # creating p random patterns with 120 bits in a 120xp matrix
        patterns = np.random.randint(0, 2, (N, p))
        patterns[patterns == 0] = -1


        weight_matrix = sum([np.dot(patterns[:, i:i + 1], patterns[:, i:i + 1].T) for i in range(p)])

        np.fill_diagonal(weight_matrix, 0)

        # Drawing a random number between 0 and 11 to choose randomly one pattern to feed
        random_pattern_index = random.randint(1, 11)
        # Taking out one pattern to be feed randomly
        feed_pattern = patterns[:, random_pattern_index]

        # Taking one bit/ neuron to update randomly
        random_bit_index = random.randint(0, 119)
        random_bit = feed_pattern[random_bit_index]

        weight = weight_matrix[random_bit_index:random_bit_index + 1, :]
        b = sum([weight.T[i] * feed_pattern[i] for i in range(N)])/N

        new_bit = np.sign(b)
        if new_bit == 0:
            new_bit = 1
        if new_bit != random_bit:
            errors += 1

    probability_error.append(errors / 10 ** 3)


print(probability_error)
print(time() - start_time)