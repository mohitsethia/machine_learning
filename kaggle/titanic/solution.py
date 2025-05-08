import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

passenger_ids = test_df['PassengerId'].copy()

# train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
# train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
# test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# this column doesn't contribute to the result, so better to remove
train_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
test_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

scaler = StandardScaler()
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

X_train = train_df.drop(columns=['Survived'])
y_train = train_df['Survived']

# Ensure all features are numeric
X_train = pd.get_dummies(X_train)

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_train_np = X_train.to_numpy(dtype=np.float64)
y_train_np = y_train.to_numpy(dtype=np.float64)

X_test = test_df
X_test_np = X_test.to_numpy(dtype=np.float64)

w_tmp  = np.zeros(X_train_np.shape[1])
b_tmp  = 0.
alph = 0.01
iters = 10000

w_out, b_out = gradient_descent(X_train_np, y_train_np, w_tmp, b_tmp, alph, iters) 
# print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

predictions = predict(X_test_np, w_out, b_out)
# print("Predictions:", predictions)
# print("Ground truth:", y_train)

# def compute_accuracy(y_pred, y_true):
#     return np.mean(y_pred == y_true)

# print("Accuracy:", compute_accuracy(predictions, y_train))

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
