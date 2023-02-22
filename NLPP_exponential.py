import numpy as np
from scipy.optimize import minimize

# Define the loss function
def loss(w):
    x1, x2 = w
    return 3*np.exp(2*x1 + 1) + 2*np.exp(x2 + 5)

# Define the constraint function
def constraint(w):
    x1, x2 = w
    return x1 + x2 - 7

# Define the Lagrangian function
def lagrangian(w, lambd):
    return loss(w) + lambd * constraint(w)

# Define the gradient of the Lagrangian function
def lagrangian_grad(w, lambd):
    x1, x2 = w
    loss_grad = np.array([6*np.exp(2*x1 + 1), 2*np.exp(x2 + 5)])
    constraint_grad = np.array([1, 1])
    return loss_grad + lambd * constraint_grad

# Set the initial guess for the weights
w0 = np.array([0, 0])

# Set the initial guess for the Lagrange multiplier
lambd0 = 0

# Set the constraint
constraint_func = constraint
constraint_dict = {'type': 'eq', 'fun': constraint_func}

# Set the convergence tolerance
tol = 1e-6

# Initialize the Lagrange multiplier and the weights
lambd = lambd0
w = w0

# Run the Lagrangian algorithm until convergence
while np.abs(constraint_func(w)) > tol:
    result = minimize(lambda x: lagrangian(x, lambd), w, jac=lambda x: lagrangian_grad(x, lambd), constraints=constraint_dict)
    w = result.x
    lambd += constraint_func(w)

# Print the optimal weights and objective value
print("Optimal weights: ", w)
print("Optimal objective value: ", loss(w))
