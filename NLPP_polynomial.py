import numpy as np
from scipy.optimize import minimize

# Define the loss function
def loss(w):
    x1, x2, x3 = w
    return x1**2 + x2**2 + x3**2

# Define the constraint functions
def constraint1(w):
    x1, x2, x3 = w
    return x1 + x2 + 3*x3 - 2
def constraint2(w):
    x1, x2, x3 = w
    return 5*x1 + 2*x2 + x3 - 5

# Define the Lagrangian function
def lagrangian(w, lambd):
    return loss(w) + lambd[0] * constraint1(w) + lambd[1] * constraint2(w)

# Define the gradient of the Lagrangian function
def lagrangian_grad(w, lambd):
    x1, x2, x3 = w
    loss_grad = np.array([2*x1, 2*x2, 2*x3])
    constraint1_grad = np.array([1, 1, 3])
    constraint2_grad = np.array([5, 2, 1])
    return loss_grad + lambd[0] * constraint1_grad + lambd[1] * constraint2_grad

# Set the initial guess for the weights
w0 = np.array([0, 0, 0])

# Set the initial guess for the Lagrange multiplier
lambd0 = np.array([0.0, 0.0])

# Set the constraints
constraint1_func = constraint1
constraint1_dict = {'type': 'eq', 'fun': constraint1_func}
constraint2_func = constraint2
constraint2_dict = {'type': 'eq', 'fun': constraint2_func}

# Set the convergence tolerance
tol = 1e-6

# Initialize the Lagrange multiplier and the weights
lambd = lambd0
w = w0

# Run the Lagrangian algorithm until convergence
while (np.abs(constraint1_func(w)) > tol) or (np.abs(constraint2_func(w)) > tol):
    result = minimize(lambda x: lagrangian(x, lambd), w, jac=lambda x: lagrangian_grad(x, lambd), constraints=[constraint1_dict, constraint2_dict])
    w = result.x
    lambd += np.array([constraint1_func(w), constraint2_func(w)])

# Print the optimal weights and objective value
print("Optimal weights: ", w)
print("Optimal objective value: ", loss(w))
