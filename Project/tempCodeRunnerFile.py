import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import padasip as pd

# Load the .mat files
path = loadmat('path.mat')['path'].ravel()
css = loadmat('css.mat')['css'].ravel()

# Create far-end signal by concatenating 10 blocks of CSS data
far_end = np.tile(css, 10)

# Feed the far-end signal into the echo path to create the echo signal
echo = np.convolve(path, far_end, mode='full')[:len(far_end)]

# Adaptive filter
filter_length = 128
mu = 0.25
eps = 1e-6
filter = pd.filters.FilterNLMS(filter_length, mu=mu, eps=eps)

# Make the far_end 2D array for the adaptive filter
far_end_padded = np.concatenate((np.zeros(filter_length - 1), far_end))
far_end_2d = np.array([far_end_padded[i: i + filter_length] for i in range(len(far_end))])

# Get the echo signal that corresponds to the far_end segments
d = echo[:len(far_end_2d)]

# Train the filter
y, e, w = filter.run(d, far_end_2d)

# Plotting
plt.figure()
plt.plot(far_end, label='Far-End Signal')
plt.plot(echo, label='Echo')
plt.plot(e, label='Error Signal')
plt.legend()
plt.show()

plt.figure()
plt.plot(path, label='Echo Path')
plt.plot(w[-1], label='Echo Path Estimate')
plt.legend()
plt.show()