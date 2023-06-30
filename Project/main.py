from scipy.io import loadmat
from scipy.signal import freqz
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


matplotlib.use('TkAgg')

# # Load the file
# mat_data = loadmat('path.mat')

# # Extract 'h' from the loaded data
# h = mat_data['path']

# # Reshape h to make it 1D
# h = h.flatten()

# # Plot the impulse response
# plt.figure()
# plt.plot(h)
# plt.title('Impulse Response')

# # Calculate and plot the frequency response using freqz
# w, H = freqz(h, worN=8000)  # Assuming 8 kHz sampling rate
# f = w / (2*np.pi) * 8000  # Convert rad/sample to Hz

# # Convert magnitude to dB
# H_db = 20 * np.log10(np.abs(H))

# plt.figure()
# plt.plot(f, H_db)
# plt.title('Frequency Response')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.show()



#################################################
# Part 2

# from scipy.io import loadmat
# from scipy.signal import welch
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the CSS file
# mat_data = loadmat('css.mat')

# # Extract 'css' from the loaded data
# css = mat_data['css']

# # Reshape css to make it 1D
# css = css.flatten()

# # Plot the samples of the CSS data
# plt.figure()
# plt.plot(css)
# plt.title('CSS Data')

# # Calculate and plot the PSD using Welch's method
# f, Pxx = welch(css, fs=8000)  # Assuming 8 kHz sampling rate
# plt.figure()
# plt.semilogy(f, Pxx)
# plt.title('Power Spectrum Density (PSD)')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.grid()
# plt.show()


##############################################################


# # part 3

# from scipy.io import loadmat
# from scipy.signal import convolve
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the CSS file
# css_data = loadmat('css.mat')
# css = css_data['css'].flatten()

# # Load the impulse response
# path_data = loadmat('path.mat')
# h = path_data['path'].flatten()

# # Concatenate five blocks of the CSS data
# css_concatenated = np.tile(css, 5)

# # Feed this into the echo path
# echo = convolve(css_concatenated, h, mode='full')

# # Plot the concatenated CSS data (far-end signal x5)
# plt.figure()
# plt.plot(css_concatenated)
# plt.title('Far-End Signal x5')

# # Plot the resulting echo signal with y-axis limits from -5 to 5
# plt.figure()
# plt.plot(echo)
# plt.title('Echo Signal')
# plt.ylim(-5, 5)

# # Estimate the input and output powers in dB
# N = len(css_concatenated)
# P_in = 10 * np.log10(1/N * np.sum(np.abs(css_concatenated)**2))
# P_out = 10 * np.log10(1/N * np.sum(np.abs(echo)**2))

# print(f'Input power: {P_in} dB')
# print(f'Output power: {P_out} dB')

# # Evaluate the echo-return-loss (ERL)
# ERL = P_in - P_out
# print(f'Echo Return Loss: {ERL} dB')

# plt.show()



##############################################################

# part 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import padasip as pd
from padasip.filters import FilterRLS
from scipy.signal import freqz



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
#plt.show()

plt.figure()
plt.plot(path, label='Echo Path')
plt.plot(w[-1], label='Echo Path Estimate')
plt.legend()
#plt.show()


##############################################################

# # part 5


# Calculate the frequency response of the estimated echo path
w_est, h_est = freqz(w[-1], worN=8000)

# Calculate the frequency response of the actual echo path
w_path, h_path = freqz(path, worN=8000)

# Calculate the amplitude responses
amp_est = 20 * np.log10(abs(h_est))
amp_path = 20 * np.log10(abs(h_path))

# Calculate the phase responses
phase_est = np.unwrap(np.angle(h_est)) 
phase_path = np.unwrap(np.angle(h_path)) 

# Plot the amplitude responses
plt.figure()
plt.plot(w_est/(2*np.pi), amp_est, label='Estimated FIR')
plt.plot(w_path/(2*np.pi), amp_path, label='Given FIR')
plt.legend()
plt.title('Amplitude Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude [dB]')
plt.grid()

# Plot the phase responses
plt.figure()
plt.plot(w_est/(2*np.pi), phase_est, label='Estimated FIR')
plt.plot(w_path/(2*np.pi), phase_path, label='Given FIR')
plt.legend()
plt.title('Phase Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase [rad]')
plt.grid()

# plt.show()

##############################################################

# part 6

# Import the RLS filter

# Create an RLS filter
filter_rls = FilterRLS(n=filter_length, mu=0.99, w="random")

# Apply the RLS filter
y_rls, e_rls, w_rls = filter_rls.run(d, far_end_2d)

# Calculate the error for ε-NLMS and RLS
error_enlms = 10 * np.log10(np.mean(np.square(e)))
error_rls = 10 * np.log10(np.mean(np.square(e_rls)))

print(f"Error for ε-NLMS: {error_enlms} dB")
print(f"Error for RLS: {error_rls} dB")

# Plot the estimated FIR filters
plt.figure()
plt.plot(w[-1], label='Estimated FIR (ε-NLMS)')
plt.plot(w_rls[-1], label='Estimated FIR (RLS)')
plt.legend()
plt.title('Estimated FIR Filters')
plt.xlabel('Tap')
plt.ylabel('Weight')
plt.grid()

plt.show()

