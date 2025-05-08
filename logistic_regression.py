import numpy as np

"""
Formulas:
m is the number of training set
n is the number of features

Sigmoid
g(z) = 1/(1+e^-z)

ƒ(w, b)(x) = g(w.x + b) = g(z) # resolves as sigmoid function
z = w.x + b

loss (ƒ(w, b)(x[i]), y[i]) = -y[i] * log (ƒ(w, b)(x[i])) - (1-y[i]) * log(1- f(w, b)(x[i]))

Cost

J(w, b) = 1/m * [i = 0, m-1]∑(loss(ƒ(w, b)(x[i]), y[i]))

Gradient Descent
repeat until convergence: {
    b = b - learning_rate_alpha * dJ(w, b)/db
    w[j] = w[j] - learning_rate_alpha * dJ(w, b)/dw[j] # for j = [0, n-1]
}

Gradient

dJ(w, b)/dw = 1/m * [i = 0, m-1] ∑ ((ƒ(w, b)(x[i]) - y[i]) * x[i])
dJ(w, b)/db = 1/m * [i = 0, m-1] ∑ (ƒ(w, b)(x[i]) - y[i])


for regularization
dJ(w, b)/dw = 1/m * [i = 0, m-1] ∑ ((f(w, b)(x[i]) - y[i]) * x[i]) + ((lambda/m) * [j = 0, n-1] ∑ w[j])

J(w, b) = 1/m * [i = 0, m-1]∑(loss(ƒ(w, b)(x[i]), y[i])) + ((lambda / (2*m)) * [j = 0, n-1] ∑ (w[j] ^ 2))


"""

def sigmoid(Z):
    g_z = 1 / (1+np.exp(-Z))
    return g_z

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range (0, m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1-y[i]) * np.log(1 - f_wb_i)
    cost /= m
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range (m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        dj_db += f_wb_i - y[i]
        dj_dw += X[i] * (f_wb_i - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    m = X.shape[0]
    w = w_in.copy()
    b = b_in
    for i in range (num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

preds = predict(X_train, w_out, b_out)
print("Predictions:", preds)
print("Ground truth:", y_train)

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

print("Accuracy:", compute_accuracy(preds, y_train))
