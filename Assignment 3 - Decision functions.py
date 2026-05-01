"""
III.   Decision functions
Create sample data using two classes that are separable linearly and implement the perceptron to determine an 
appropriate decision function.
Repeat this process and include data samples that will make the two classes not linearly separable and 
see if the Perceptron still works albeit with a lesser classification accuracy.
Implement the generalized decision function formulation (GDF) for the rth order and with n Illustrate that 
your program works for the cases:
(a) Second degree with 2 dimensions
(b) Third degree with 2 dimensions
(c) Second degree with 3 dimensions
(d) Third degree with 3 dimensions
Perhaps initially you set all the weights to 1 and observe your results, then as you vary 
the different weights singularly, look at the change experienced in the various GDFs.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

np.random.seed(42)

def perceptron(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = 1 if np.dot(w, xi) + b >= 0 else -1
            if pred != yi:
                w += lr * yi * xi
                b += lr * yi
                errors += 1
        if errors == 0:
            break
    return w, b

def predict(X, w, b):
    return np.where(X @ w + b >= 0, 1, -1)

def accuracy(X, y, w, b):
    return np.mean(predict(X, w, b) == y) * 100

def plot_boundary(X, y, w, b, title):
    plt.figure()
    plt.scatter(X[y == 1, 0],  X[y == 1, 1],  color='blue', label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red',  label='Class -1')
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'g-', label='Decision Boundary')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

X_pos = np.random.randn(50, 2) + [3, 3]
X_neg = np.random.randn(50, 2) + [-3, -3]
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(50), -np.ones(50)])

w, b = perceptron(X, y)
print(f"Part 1 (Separable)     Accuracy: {accuracy(X, y, w, b):.1f}%")
plot_boundary(X, y, w, b, "Perceptron — Linearly Separable")

X2 = np.vstack([
    np.random.randn(30, 2) + [2,  2],
    np.random.randn(30, 2) + [-2, -2],
    np.random.randn(30, 2) + [2, -2],
    np.random.randn(30, 2) + [-2,  2],
])
y2 = np.hstack([np.ones(60), -np.ones(60)])

w2, b2 = perceptron(X2, y2)
print(f"Part 2 (Non-Separable) Accuracy: {accuracy(X2, y2, w2, b2):.1f}%")
plot_boundary(X2, y2, w2, b2, "Perceptron — NOT Linearly Separable")

def poly_features(X, degree):
    cols = [np.ones(len(X))]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            term = np.ones(len(X))
            for i in combo:
                term *= X[:, i]
            cols.append(term)
    return np.column_stack(cols)

def gdf_predict(phi, w):
    return np.where(phi @ w >= 0, 1, -1)

def gdf_accuracy(phi, w, y):
    return np.mean(gdf_predict(phi, w) == y) * 100

def plot_gdf_2d(X, y, degree, w, title):
    plt.figure()
    plt.scatter(X[y == 1,  0], X[y == 1,  1], color='blue', label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red',  label='Class -1')
    g1 = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 300)
    g2 = np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 300)
    G1, G2 = np.meshgrid(g1, g2)
    Z = (poly_features(np.c_[G1.ravel(), G2.ravel()], degree) @ w).reshape(G1.shape)
    plt.contourf(G1, G2, Z, levels=[-1e9, 0, 1e9], colors=['#ffcccc','#ccccff'], alpha=0.4)
    plt.contour(G1, G2, Z, levels=[0], colors='green')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_gdf(label, X, y, degree):
    phi   = poly_features(X, degree)
    n_w   = phi.shape[1]
    w_all = np.ones(n_w)
    print(f"\n{'='*55}")
    print(f"GDF {label}  |  degree={degree}, dim={X.shape[1]},  {n_w} features")
    print(f"  All weights = 1  →  Accuracy: {gdf_accuracy(phi, w_all, y):.1f}%")

    print(f"  {'Weight #':<10} {'w=0':>7} {'w=0.5':>7} {'w=2':>7} {'w=5':>7} {'w=-1':>7}")
    for i in range(n_w):
        row = f"  w[{i}]      "
        for val in [0.0, 0.5, 2.0, 5.0, -1.0]:
            w_test    = np.ones(n_w)
            w_test[i] = val
            row += f" {gdf_accuracy(phi, w_test, y):>6.1f}%"
        print(row)

    return phi, w_all

Xg2 = np.random.randn(120, 2) * 2
yg2 = np.where(Xg2[:, 0]**2 + Xg2[:, 1]**2 <= 4, 1, -1)

Xg3 = np.random.randn(120, 3) * 2
yg3 = np.where(Xg3[:, 0]**2 + Xg3[:, 1]**2 + Xg3[:, 2]**2 <= 4, 1, -1)

phi_a, w_a = run_gdf("(a)", Xg2, yg2, degree=2)
plot_gdf_2d(Xg2, yg2, 2, w_a, "GDF (a): 2nd degree, 2D all weights=1")

phi_b, w_b = run_gdf("(b)", Xg2, yg2, degree=3)
plot_gdf_2d(Xg2, yg2, 3, w_b, "GDF (b): 3rd degree, 2D all weights=1")

phi_c, w_c = run_gdf("(c)", Xg3, yg3, degree=2)

phi_d, w_d = run_gdf("(d)", Xg3, yg3, degree=3)