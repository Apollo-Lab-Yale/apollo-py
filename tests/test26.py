import numpy as np

# Example: n x h matrix and n x 1 vector
A = np.random.rand(4, 5)  # 4 x 5 matrix
v = np.random.rand(4, 1)  # 4 x 1 vector

# Subtract v from each column of A
distances = np.linalg.norm(A - v, axis=0)

print(distances)
