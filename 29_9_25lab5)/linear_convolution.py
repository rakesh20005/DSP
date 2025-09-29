import numpy as np
import matplotlib.pyplot as plt

def manual_dft(x):
    # Manually computes the Discrete Fourier Transform (DFT).
    N = x.size
    X_real = np.zeros(N)
    X_imag = np.zeros(N)
    
    for k in range(N):
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            # DFT formula split into real and imaginary parts.
            X_real[k] += x[n] * np.cos(angle)
            X_imag[k] -= x[n] * np.sin(angle)
    return X_real, X_imag

def manual_idft(X_real, X_imag):
    # Manually computes the Inverse DFT (IDFT).
    N = X_real.size
    x_reconstructed = np.zeros(N, dtype=np.complex128)
    
    for n in range(N):
        real_sum = 0
        imag_sum = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            # IDFT formula split into real and imaginary parts.
            real_sum += X_real[k] * np.cos(angle) - X_imag[k] * np.sin(angle)
            imag_sum += X_real[k] * np.sin(angle) + X_imag[k] * np.cos(angle)
        x_reconstructed[n] = (real_sum + 1j * imag_sum) / N
    # Return only the real part of the result.
    return np.real(x_reconstructed)

# Define two sequences
x = np.array([10, 70, 20, 40, 60, 90, 100])
h = np.array([1, 1, 1, 1, 1, 1, 1, 10, 70, 20, 40])

# Calculate length of the convolved signal
N = len(x) + len(h) - 1

# Zero-pad sequences to length N
x_padded = np.pad(x, (0, N - len(x)))
h_padded = np.pad(h, (0, N - len(h)))

# Compute DFT of padded sequences
X_real, X_imag = manual_dft(x_padded)
H_real, H_imag = manual_dft(h_padded)

# Multiply DFTs in the frequency domain (complex multiplication)
Y_real = X_real * H_real - X_imag * H_imag
Y_imag = X_real * H_imag + X_imag * H_real

# Compute IDFT to get the convolution result
y = manual_idft(Y_real, Y_imag)

# Use numpy's built-in convolution for comparison
y_conv = np.convolve(x, h)

# Plotting the results
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.stem(np.arange(len(x)), x)
plt.title('Input Vector x')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xticks(np.arange(len(x)))
plt.grid(True)

plt.subplot(2, 2, 2)
plt.stem(np.arange(len(h)), h)
plt.title('Input Vector h')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xticks(np.arange(len(h)))
plt.grid(True)

plt.subplot(2, 2, 3)
plt.stem(np.arange(len(y)), y)
plt.title('Convolution using Manual DFT')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xticks(np.arange(len(y)))
plt.grid(True)

plt.subplot(2, 2, 4)
plt.stem(np.arange(len(y_conv)), y_conv)
plt.title('Convolution using np.convolve')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xticks(np.arange(len(y_conv)))
plt.grid(True)

plt.tight_layout()
plt.show()

print("Linear convolution result (manual DFT):", y)
print("Linear convolution result (np.convolve):", y_conv)
