import math
from numpy import add, subtract, dot, random, zeros
from numpy.linalg import inv, norm, eig
import matplotlib.pyplot as plt
import numpy as np


def f(x, Q, b):
    val = add(0.5 * dot(dot(x.T, Q), x), dot(b.T, x))
    return val[0][0]


def gradF(x, Q, b):
    return add(dot(Q, x), b)


def projection(x_vector):
    for i in range(len(x_vector)):
        if x_vector[i] < 0:
            x_vector[i] = 0
        elif x_vector[i] > 1:
            x_vector[i] = 1

    return x_vector


n = 100
B = zeros(shape=(n, n))
D = zeros(shape=(n, n))
b = zeros(shape=(n, 1))  # vector
# initialize B
for i in range(n):
    for j in range(n):
        B[i, j] = random.normal(loc=0.0, scale=1.0)

# initialize D
for i in range(n):
    D[i, i] = random.lognormal()

for i in range(n):
    b[i, 0] = random.normal(loc=0.0, scale=10)

Q = add(dot(B, B.T), D)
eigenValues, eigenVectors = eig(Q)
L = max(eigenValues)
m = min(eigenValues)
condition_number = L / m

X_optimal = dot(inv(Q), -b)

X_0 = zeros(shape=(n, 1))
iterations = 2000
X = X_0

# 1: constant step-size:
X = X_0
alpha = 3 / L
epsilon_list = []
for i in range(iterations):
    X = subtract(X, alpha * gradF(X, Q, b))
    X = projection(X)
    epsilon = pow(norm(X - projection(X - gradF(X, Q, b))), 2)
    epsilon_list.append(math.log(epsilon))

print("Epsilon for constant step-size: ", math.log(epsilon))
plt.scatter(range(1, iterations + 1), epsilon_list)
plt.show()

# 2: diminishing:
X = X_0
alpha = 100 / L
r = 1
epsilon_list = []
for i in range(iterations):
    X = subtract(X, (alpha * math.log(r) / r) * gradF(X, Q, b))
    X = projection(X)
    r += 1
    epsilon = pow(norm(X - projection(X - gradF(X, Q, b))), 2)
    epsilon_list.append(math.log(epsilon))

print("Epsilon for diminishing step size:", math.log(epsilon))
plt.scatter(range(1, iterations + 1), epsilon_list)
plt.show()

# 3: Nesterov's Acceleration

iteration_numbers = 200
a = [0.0]
t = []
for i in range(1, iteration_numbers+1):
    new_a = 1 / 2 * (1 + math.sqrt(4 * a[i-1] * a[i-1] + 1))
    a.append(new_a)

    t_value = (a[i-1] - 1) / (a[i])
    t.append(t_value)

print(a)
print(t)

epsilon_list = []
X_old = np.zeros(shape=(n, 1))
X_new = np.zeros(shape=(n, 1))
for i in range(iteration_numbers):
    Y = (1 + t[i]) * X_new - t[i] * X_old
    X_old = X_new.copy()
    X_new = Y - 1 / L * gradF(Y, Q, b)
    X_new = projection(X_new)
    epsilon = pow(norm(X_new - projection(X_new - gradF(X_new, Q, b))), 2)
    epsilon_list.append(math.log(epsilon))


print("Epsilon for Aggressive Nesterov:", math.log(epsilon))
plt.scatter(range(1, iteration_numbers + 1), epsilon_list)
plt.show()
