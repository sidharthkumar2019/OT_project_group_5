import numpy as np
from scipy.optimize import minimize

# Define the loss function
def loss(w):
    return w[0]**2 + w[1]**2 + w[2]**2

# Define the constraints
def constraint1(w):
    return w[0] + w[1] + 3*w[2] - 2

def constraint2(w):
    return 5*w[0] + 2*w[1] + w[2] - 5

# Define the Lagrangian function
def lagrangian(w, lambd):
    return loss(w) + np.sum([lambd[i] * constraint_funcs[i](w) for i in range(num_constraints)])

# Define the gradient of the Lagrangian function
def lagrangian_grad(w, lambd):
    loss_grad = 2 * w
    constraint_grads = np.array([[1, 1, 3],
                                 [5, 2, 1]])
    return loss_grad + np.sum([lambd[i] * constraint_grads[i] for i in range(num_constraints)], axis=0)

# Set the constraints
constraint_funcs = [constraint1, constraint2]
num_constraints = len(constraint_funcs)
constraints = [{'type': 'eq', 'fun': f} for f in constraint_funcs]

# Set the initial guess for the weights
w0 = np.array([0, 0, 0])

# Set the initial guess for the Lagrange multipliers
lambd0 = np.zeros(num_constraints)


# Set the convergence tolerance
tol = 1e-6

# Initialize the Lagrange multipliers and the weights
lambd = lambd0
w = w0

# Run the Lagrangian algorithm until convergence
while np.max(np.abs([f(w) for f in constraint_funcs])) > tol:
    result = minimize(lambda x: lagrangian(x, lambd), w, jac=lambda x: lagrangian_grad(x, lambd), constraints=constraints)
    w = result.x
    lambd += np.array([f(w) for f in constraint_funcs])

# Print the optimal weights
print("Optimal weights: ", w)
