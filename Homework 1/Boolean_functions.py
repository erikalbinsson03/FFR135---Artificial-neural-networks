import numpy as np
from itertools import product

NUMBER_OF_RUNS = 20
NUMBER_OF_EPOCHS = 20

def generate_all_inputs(n):
    inputs = list(product([-1, 1], repeat=n))

    return np.array(inputs)


def generate_all_functions(n):
    number_of_inputs = 2 ** n
    functions = list(product([-1, 1], repeat=number_of_inputs))

    return np.array(functions)


def train_perceptron(inputs, function, n):
    weights = np.random.normal(0, 1/np.sqrt(n), n)
    eta = 0.05
    theta = 0

    for epoch in range(NUMBER_OF_EPOCHS):
        for i in range(len(inputs)):
            x = inputs[i]
            t = function[i]

            b = np.dot(weights, x) - theta
            O = np.where(b >= 0, 1, -1)

            weights += eta * (t - O) * x
            theta += -eta * (t - O)

    return weights, theta


def is_linearly_separable(inputs, function, n):
    weights, theta = train_perceptron(inputs, function, n)

    for i in range(len(inputs)):
        x = inputs[i]
        t = function[i]

        b = np.dot(weights, x) - theta
        O = np.where(b >= 0, 1, -1)

        if O != t:
            return False
    return True

# n = 2 and 3
for n in [2, 3]:
    inputs = generate_all_inputs(n)
    functions = generate_all_functions(n)
    fractions_linearly_separable = []

    for run in range(NUMBER_OF_RUNS):
        count_linearly_separable = 0

        for f in functions:
            if is_linearly_separable(inputs, f, n):
                count_linearly_separable += 1

        fraction = count_linearly_separable / len(functions)
        fractions_linearly_separable.append(fraction)

    print(f"n = {n}")
    print(f"Number of boolean functions considered: {len(functions)}")
    print(f"Average fraction of linear seperability: {np.mean(fractions_linearly_separable)}")
    print(f"Standard deviaton: {np.std(fractions_linearly_separable)}")

# n = 4 and 5
NUMBER_OF_SAMPLED_FUNCTIONS = 10 ** 4

def generate_random_functions(n, number_of_functions):
    number_of_inputs = 2 ** n
    sampled_functions = np.random.choice([-1, 1], size=(number_of_functions, number_of_inputs))

    return sampled_functions


for n in [4, 5]:
    inputs = generate_all_inputs(n)
    fractions_linearly_separable = []

    for run in range(NUMBER_OF_RUNS):
        functions = generate_random_functions(n, NUMBER_OF_SAMPLED_FUNCTIONS)
        count_linearly_separable = 0

        for f in functions:
            if is_linearly_separable(inputs, f, n):
                count_linearly_separable += 1

        fraction = count_linearly_separable / len(functions)
        fractions_linearly_separable.append(fraction)

    print(f"n = {n}")
    print(f"Number of boolean functions sampled: {NUMBER_OF_SAMPLED_FUNCTIONS}")
    print(f"Average fraction of linear separability: {np.mean(fractions_linearly_separable)}")
    print(f"Standard deviation: {np.std(fractions_linearly_separable)}")



    
