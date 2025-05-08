import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

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

def generate_polynomial_features(X, degree=2):
    m, n = X.shape
    features = [X]  # original features

    if degree >= 2:
        # Add squared terms
        squared = X ** 2
        features.append(squared)

        # Add interaction terms: x_i * x_j where i < j
        interactions = []
        for i in range(n):
            for j in range(i + 1, n):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                interactions.append(interaction)
        if interactions:
            features.append(np.hstack(interactions))

    return np.hstack(features)

def predict(x, w, b):
    return np.dot(x, w) + b

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop Id, separate target
train_df = train_df.drop(columns=["Id"])
X_train = train_df.drop(columns=["SalePrice"])
y_train = train_df["SalePrice"]

# Combine with test for consistent encoding
test_ids = test_df["Id"]
X_all = pd.concat([X_train, test_df.drop(columns=["Id"])], axis=0)

# Fill missing values
X_all = X_all.fillna(X_all.mode().iloc[0])

# One-hot encode first
X_all_encoded = pd.get_dummies(X_all)

selector = VarianceThreshold(threshold=0.01)  # adjust as needed
X_all_var_filtered = selector.fit_transform(X_all_encoded)

# Convert to NumPy and ensure float type
# X_all_array = X_all_var_filtered.astype(np.float64).to_numpy()
X_all_array = X_all_var_filtered.astype(np.float64)

# Optional: Add small epsilon to avoid log(0)
X_all_array = np.log1p(X_all_array + 1e-8)

X_all_poly = generate_polynomial_features(X_all_array, degree=2)

X_all_poly = np.clip(X_all_poly, -1e5, 1e5)

# Standardize
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all_poly)

# Re-split into train and test
X_train_final = X_all_scaled[:len(X_train), :]
X_test_final = X_all_scaled[len(X_train):, :]

initial_w = np.zeros(X_train_final.shape[1])
initial_b = 0
learning_rate_alpha = 0.001
iterations = 10000

w_final, b_final = gradient_descent(X_train_final, y_train.to_numpy(), initial_w, initial_b, learning_rate_alpha, iterations, compute_cost, compute_gradient)

predictions = predict(X_test_final, w_final, b_final)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})
submission.to_csv("submission.csv", index=False)

# print(f"Learned weight (w): {w_final}")
# print(f"Learned bias (b): {b_final:8.4f}")
# print(f"predicted prices for the areas {X_test_final}: {predictions}")
