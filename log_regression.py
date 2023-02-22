import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return  1 / (1 + np.exp(-z))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def logistic_regression(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(np.dot(X, theta))
        gradient_val = gradient(X, h, y)
        theta = theta - (gradient_val * alpha)
        
    return theta
  
data = pd.read_csv('irisdata.csv')
  
X = data[['length', 'width']]
y = data['Type']

theta = np.zeros(X.shape[1])

alpha = 0.01
iterations = 20000

h = sigmoid(np.dot(X, theta))

theta =logistic_regression(X, y, theta, alpha, iterations)

plt.scatter(X.length,X.width)
plt.show()

print(sigmoid(np.dot([6, 2],theta)))
