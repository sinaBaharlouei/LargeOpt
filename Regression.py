from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# X, y = load_boston(return_X_y=True)
X, y = load_diabetes(return_X_y=True)


# X = X[:, np.newaxis, 2]

# plt.scatter(X, y)
# plt.show()

ones = np.ones(X.shape[0])
ones = ones[:, np.newaxis]

X1 = np.concatenate([X, ones], 1)

y = y[:, np.newaxis]

trainX = X1[:400]
testX = X1[400:]

trainY = y[:400]
testY = y[400:]

lam = 0
# TODO: Implement Linear Regression
theta_star = np.dot(np.linalg.inv(np.dot(trainX.T, trainX)), np.dot(trainX.T, trainY))
predicted_Y = np.dot(testX, theta_star)


# TODO: Implement Linear Regression via GD
theta = np.zeros(X1.shape[1])
theta = theta[:, np.newaxis]
alpha = 0.002
print(theta.shape)

# y_gd = np.dot(testX, theta)
# plt.scatter(predicted_Y, testY)
# plt.show()

standard_deviation = np.std(testY)

rmse = np.sqrt(1 / testY.shape[0] * np.linalg.norm(testY - predicted_Y)**2)
normalized_rmse = rmse / standard_deviation
print("Closed form solution NRMSE: ", normalized_rmse)

#
# rmse = np.sqrt(1 / testY.shape[0] * np.linalg.norm(testY - y_gd)**2)
# normalized_rmse = rmse / standard_deviation
# print("GD NRMSE: ", normalized_rmse)
