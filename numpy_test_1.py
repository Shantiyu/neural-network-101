import numpy as np

# This is a simple sample script that tries out some basic use of Numpy (np).

# Numpy supports high-performance batched calculation for array, matrix and tensor.
# This is based on its c/cpp backend.

# Declaration of two arrays (or, two vectors / 1D Tensors)
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 3, 5, 7, 9])

# Numpy supports convenient calculation for arrays.
print(a + b)
print(a - b)
print(a * b) # Element-wise product
print(a / b)
print(a + 1) # Add each element in a by 1
print(a * 2)
print(b[0]) # Access the array

# Diving by zero will NOT throw error in numpy. Instead, it throws a RuntimeWarning, and assign the result as inf.
b[0] = 0
print(a / b) # result : [inf 0.66666667 0.6 0.57142857 0.55555556]