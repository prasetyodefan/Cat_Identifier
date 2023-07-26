import numpy as np

# Create a large NumPy array
large_array = np.arange(1000)

# Without setting the threshold
print(large_array)

# Set the threshold to infinity to print the whole array
np.set_printoptions(threshold=np.inf)
print(large_array)
