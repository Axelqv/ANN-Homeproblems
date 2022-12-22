import numpy as np
import random
from time import time

# The nr of patterns
p_vector = [12, 24, 48, 70, 100, 120]
# the number of bits
N = 120
# Creating a errors variable that will grow when errors occur( when updated_bit != bit)
start_time = time()
probability_error = []
for p in p_vector:
    errors = 0
    for _ in range(10**3):

        # creating p random patterns with 120 bits in a px120 matrix
        patterns = np.random.randint(0, 2, (p, N))
        patterns[patterns == 0] = -1


        # Taking one bit/ neuron to update randomly
        random_bit = random.randint(0, N-1)

        # list with all the weights for the randomly chosen bit
        weights = [sum([patterns[my, random_bit] * patterns[my, j] for my in range(p)]) for j in range(N)]

        # Drawing a random number between 0 and p to choose randomly one pattern to feed
        random_pattern_index = random.randint(0, p-1)
        # Taking out one pattern to be feed randomly
        feed_pattern = patterns[random_pattern_index, :]

        # Calculating the local field b
        b = [weights[j] * feed_pattern[j] for j in range(N) if j!= random_bit]
        b = sum(b) / N

        if b == 0:
            b = 1

        new_bit = np.sign(b)  # Updating the bit
        if new_bit != feed_pattern[random_bit]:
            errors += 1

    probability_error.append(errors/(10**3))

print(probability_error)
print(time() - start_time)