import numpy as np
import math

"""
Formulas

# cost function
ƒ(w, b)(x[i]) = w.x[i] + b
J(w, b)[i] = 1/(2*m) * [i = 0, m-1]∑ ((ƒ(w, b)(x[i]) - y[i]) ^ 2) = [i = 0, m-1] ∑ ((w.x[i] + b - y[i])^2)

# gradient
dJ(w, b)/dw = 1/m * [i = 0, m-1]∑(ƒ(w, b)(x[i]) - y[i]) * x[i]
dJ(w, b)/db = 1/m * [i = 0, m-1]∑(ƒ(w, b)(x[i]) - y[i])

#gradient descent
repeat until convergence: {
    b = b - a * dJ(w, b)/db
    w = w - a * dJ(w, b)/dw
}

Implement Gradient Descent
To implement gradient descent algorithm for one feature. You will need three functions.

compute_gradient
compute_cost
gradient_descent, utilizing compute_gradient and compute_cost

for regularization

J(w, b)[i] = 1/(2*m) * [i = 0, m-1]∑ ((ƒ(w, b)(x[i]) - y[i]) ^ 2) = [i = 0, m-1] ∑ ((w.x[i] + b - y[i])^2) + ((lambda / (2*m)) * [j = 0, n-1] ∑ (w[j] ^ 2))

dJ(w, b)/dw = 1/m * [i = 0, m-1]∑(ƒ(w, b)(x[i]) - y[i]) * x[i] + ((lambda/m) * [j = 0, n-1] ∑ w[j])

"""

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range (0, m):
        f_wb = np.dot(x[i], w) + b
        cost_sum += (f_wb - y[i]) ** 2
    total_cost = cost_sum / (2*m)
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_db = 0
    dj_dw = np.zeros(w.shape[0])
    for i in range (0, m):
        f_wb = np.dot(x[i], w) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_func, gradient_func):
    J_history = []
    p_history = []
    b = b_in
    w = w_in.copy()
    for i in range (num_iters):
        dj_dw, dj_db = gradient_func(x, y, w, b)
        b = b - alpha * (dj_db)
        w = w - alpha * (dj_dw)
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_func(x, y, w , b))
            p_history.append([w.copy(),b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w}, b:{b: 0.5e}")
    return w, b

def predict(x, w, b):
    return np.dot(x, w) + b

# Load our data set
# Example: training data with 2 features (e.g., [area, number_of_rooms])
x_train = np.array([
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0]
])  # shape (4, 2)

y_train = np.array([300.0, 400.0, 700.0, 900.0])  # shape (4,)

initial_w = np.zeros(x_train.shape[1])
initial_b = 0
learning_rate_alpha = 0.01
iterations = 10000

w_final, b_final = gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate_alpha, iterations, compute_cost, compute_gradient)

# Predicting prices for new examples
x_test = np.array([
    [1.5, 1.0],
    [2.5, 1.5],
    [3.5, 2.5]
])  # shape (3, 2)

predictions = predict(x_test, w_final, b_final)

print(f"Learned weight (w): {w_final}")
print(f"Learned bias (b): {b_final:8.4f}")
print(f"predicted prices for the areas {x_test}: {predictions}")
