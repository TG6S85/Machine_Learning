# Implimentation of a simple machine learning perceptron model AND
# randomized linearly seperable data set generator.
##################################################################

import numpy as np
import matplotlib.pyplot as plt


# So this function is the main guesser, literally predicts what g(x) is
def predict(xa, xb, y, weights):
    act = weights[0]
    act += weights[1] * xa
    act += weights[2] * xb
    return 1 if act >= 0 else 0


# Uses the predict func on test data to update weights
def perceptron(x1, x2, y, l_rate, n_epoch):
    weights = [1, 1, 1]  # Starting weight values, just random.
    for epoch in range(n_epoch):  # Run through this entire op for n in epoch
        sum_error = 0
        for i in range(1000):  # for every value in the dataset
            pre = predict(x1[i], x2[i], y[i], weights)  # Run prediction
            error = y[i] - pre
            sum_error += error
            # Updating weights
            weights[0] = weights[0] + l_rate * error
            weights[2] = weights[2] + l_rate * error * x2[i]
            weights[1] = weights[1] + l_rate * error * x1[i]
        # Printing the epoch's training statistics
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        s = -(weights[0]/weights[2])/(weights[0]/weights[1])  # Slope
        bin = -weights[0]/weights[2]  # intercept
        # plt.plot(x1, s*x1+bin, ':')
    return weights


x1 = np.array([])
x2 = np.array([])
# time for creating the target func
i = 0
y = np.zeros((1000))

x1 = np.random.rand(1000, 1)
x2 = np.random.rand(1000, 1)
# this is the target func
for i in range(1000):
    x1[i] *= 15
    x2[i] *= 25
    if 2*x1[i] > x2[i]:
        y[i] = 1
        plt.plot(x1[i], x2[i], "ro", color='green')
    else:
        y[i] = 0
        plt.plot(x1[i], x2[i], "ro", color='red')

weights = perceptron(x1, x2, y, .1, 10)
print(weights)
s = -(weights[0]/weights[2])/(weights[0]/weights[1])
bin = -weights[0]/weights[2]
plt.plot(x1, 2*x1, color="pink", label="target")
plt.plot(x1, s*x1+bin, color="black")
plt.legend(loc='upper center')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
