import numpy as np
import matplotlib.pyplot as plt

def manual_dft(x):
    N = x.size
    X_real = np.zeros(N)
    X_imag = np.zeros(N)
    
    for k in range(N):
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            X_real[k] += x[n] * np.cos(-angle)
            X_imag[k] += x[n] * np.sin(-angle)
    return X_real, X_imag

def manual_idft(X_real, X_imag):
    N = X_real.size
    x_reconstructed = np.zeros(N)
    
    for n in range(N):
        real_sum = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            real_sum += X_real[k] * np.cos(angle) - X_imag[k] * np.sin(angle)
        x_reconstructed[n] = real_sum / N
    return x_reconstructed

x =np.array([1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15, 6,7,8,9,10,11,12,13,14])
N = x.size 

X_real, X_imag = manual_dft(x)

# Magnitude of DFT
magnitude = np.sqrt(X_real**2 + X_imag**2)

# Phase of DFT
phase = np.arctan2(X_imag, X_real)

# IDFT (reconstructed signal)
x_reconstructed = manual_idft(X_real, X_imag)

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.stem(np.arange(N), x)
plt.title("Original sequence x[n]")
plt.xlabel("n (time index)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(N))
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(np.arange(N), magnitude)
plt.title("DFT Magnitude Spectrum |X[k]|")
plt.xlabel("k (frequency bin)")
plt.ylabel("Magnitude")
plt.xticks(np.arange(N))
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(np.arange(N), phase)
plt.title("DFT Phase Spectrum ∠X[k]")
plt.xlabel("k (frequency bin)")
plt.ylabel("Phase (radians)")
plt.xticks(np.arange(N))
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(np.arange(N), x_reconstructed)
plt.title("Reconstructed sequence from IDFT")
plt.xlabel("n (time index)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(N))
plt.grid()

plt.tight_layout()
plt.show()
