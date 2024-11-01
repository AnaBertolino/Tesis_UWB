import numpy as np
import matplotlib.pyplot as plt

# Load your NumPy array
array = np.load('raw_data_0.npy')  # replace with your file path
phase = np.load('phase_data_0.npy')
phase = phase[0:500]

downsampled_array = array
downsampled_array = downsampled_array[0:500]

'''
# Plot the array
plt.figure(figsize=(8, 6))
plt.plot(downsampled_array)  # If 1D array; use plt.imshow(array, cmap='viridis') for 2D
plt.title("Array Plot")
plt.xlabel("Index")
plt.ylabel("Values")
plt.show()
'''
# Plot the array
plt.figure(figsize=(8, 6))
plt.plot(downsampled_array)  # If 1D array; use plt.imshow(array, cmap='viridis') for 2D
plt.title("Array Plot")
plt.xlabel("Index")
plt.ylabel("Values")
plt.show()