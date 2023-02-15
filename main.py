from scipy.optimize import minimize

# Define the function f(x,y,z) to be optimized
def f(xyz):
    x, y, z = xyz
    return x**2 + y**2 + z**2

# Define the constraint functions g1(x,y,z) = c1 and g2(x,y,z) = c2
def g1(xyz):
    x, y, z = xyz
    return x + y + z - 1

def g2(xyz):
    x, y, z = xyz
    return x**2 + y**2 + z**2 - 2

# Define the Lagrangian function L(x,y,z,λ1,λ2)
def lagrangian(xyz_lambda):
    x, y, z, lambda1, lambda2 = xyz_lambda
    return f([x, y, z]) - lambda1 * g1([x, y, z]) - lambda2 * g2([x, y, z])

# Define the constraints
def constraint1(xyz):
    return g1(xyz)

def constraint2(xyz):
    return g2(xyz)

# Use Scipy's minimize function with constraints to find the stationary point
result = minimize(lagrangian, [0, 0, 0, 1, 1], constraints=[{'type': 'eq', 'fun': constraint1},
                                                         {'type': 'eq', 'fun': constraint2}])

# Print the result
print(result)

