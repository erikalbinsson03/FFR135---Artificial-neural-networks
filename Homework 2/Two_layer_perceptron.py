import pandas as pd
import numpy as np

training_set = pd.read_csv('training_set.csv', header=None)
validation_set = pd.read_csv('validation_set.csv', header=None)

M1 = 5
M2 = 10
EPOCHS = 50
LEARNING_RATE = 0.01

W1 = np.random.randn(2, M1)
W2 = np.random.randn(M1, M2)
W3 = np.random.randn(M2, 1)

theta1 = np.zeros(M1)
theta2 = np.zeros(M2)
theta3 = np.zeros(1)

# training
for epoch in range(EPOCHS):
    X_train = training_set.copy()
    X_train = X_train.sample(frac=1).reset_index(drop=True)
    for input in X_train.to_numpy():
        x = input[:2]
        t = input[2]

        V1 = np.tanh(np.dot(x, W1) - theta1)
        V2 = np.tanh(np.dot(V1, W2) - theta2)
        O = np.tanh(np.dot(V2, W3) - theta3)

        delta3 = (t - O) * (1 - O ** 2)
        delta2 = delta3 * W3[:, 0] * (1 - V2 ** 2)
        delta1 = (W2 @ delta2) * (1 - V1 ** 2)

        W3 += LEARNING_RATE * np.outer(V2, delta3)
        theta3 -= LEARNING_RATE * delta3
        W2 += LEARNING_RATE * np.outer(V1, delta2)
        theta2 -= LEARNING_RATE * delta2
        W1 += LEARNING_RATE * np.outer(x, delta1)
        theta1 -= LEARNING_RATE * delta1

    np.savetxt('w1.csv', W1.T, delimiter=',', fmt='%.18g')
    np.savetxt('w2.csv', W2.T, delimiter=',', fmt='%.18g')
    np.savetxt('w3.csv', W3, delimiter=',', fmt='%.18g')
    np.savetxt('t1.csv', theta1.reshape(-1, 1), delimiter=',', fmt='%.18g')
    np.savetxt('t2.csv', theta2.reshape(-1, 1), delimiter=',', fmt='%.18g')
    np.savetxt('t3.csv', theta3.reshape(-1, 1), delimiter=',', fmt='%.18g')

# testing
predictions = []
for input in validation_set.to_numpy():
    x = input[:2]

    V1 = np.tanh(np.dot(x, W1) - theta1)
    V2 = np.tanh(np.dot(V1, W2) - theta2)
    O = np.tanh(np.dot(V2, W3) - theta3)

    predictions.append(np.sign(O))

predictions = np.array(predictions).flatten()
targets = validation_set.iloc[:, 2].to_numpy()
print(predictions.shape)
print(targets.shape)

print(f"M1 = {M1} and M2 = {M2}")
print(f"The fraction of incorrectly classified inputs is: {np.mean(predictions != targets)}")

