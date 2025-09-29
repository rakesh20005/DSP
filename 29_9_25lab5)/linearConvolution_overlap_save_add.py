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
            real_sum += X_real[k] * np.cos(angle) - X_imag[k] * np.sin(angle)
            imag_sum += X_real[k] * np.sin(angle) + X_imag[k] * np.cos(angle)
        x_reconstructed[n] = (real_sum + 1j * imag_sum) / N
    return np.real(x_reconstructed)

def dft_conv(x_block, h):
    # Performs linear convolution of a block using DFT.
    N_conv = len(x_block) + len(h) - 1
    
    x_padded = np.pad(x_block, (0, N_conv - len(x_block)))
    h_padded = np.pad(h, (0, N_conv - len(h)))
    
    X_real, X_imag = manual_dft(x_padded)
    H_real, H_imag = manual_dft(h_padded)
    
    Y_real = X_real * H_real - X_imag * H_imag
    Y_imag = X_real * H_imag + X_imag * H_real
    
    y = manual_idft(Y_real, Y_imag)
    return y

def overlap_add(x, h, L):
    # Performs convolution using the Overlap-Add method.
    M = len(h)
    N = L + M - 1
    output_length = len(x) + M - 1
    y = np.zeros(output_length)
    
    num_blocks = int(np.ceil(len(x) / L))
    
    for i in range(num_blocks):
        start_idx = i * L
        end_idx = start_idx + L
        x_block = x[start_idx:end_idx]
        
        y_block = dft_conv(x_block, h)
        
        # Add to the output buffer, handling the overlap
        overlap_start = i * L
        y[overlap_start : overlap_start + N] += y_block
        
    return y

def overlap_save(x, h, L):
    # Performs convolution using the Overlap-Save method.
    M = len(h)
    N = L + M - 1
    output = []
    
    # Pad input signal with M-1 zeros at the beginning for the first block
    x_padded_os = np.pad(x, (M - 1, 0))
    
    num_blocks = int(np.ceil((len(x) + M - 1) / L))
    
    for i in range(num_blocks):
        start_idx = i * L
        end_idx = start_idx + N
        
        if end_idx > len(x_padded_os):
            x_block = x_padded_os[start_idx:]
            x_block = np.pad(x_block, (0, N - len(x_block)))
        else:
            x_block = x_padded_os[start_idx:end_idx]
        
        y_block = dft_conv(x_block, h)
        
        # Save the "good" part (discard first M-1 samples)
        output.extend(y_block[M-1:])
    
    return np.array(output[:len(x) + M - 1])

# --- Main Script ---
# Define signals (long x for practical use of these methods)
x = np.array([512, 70, 530, 40, 60, 90, 100, 20, 80, 110, 150, 180, 200, 250, 300, 350, 400, 450, 500, 550])
h = np.array([1, 1, 1, 1, 1, 1, 1])

L = 5  # Block size for Overlap-Add (must be > 0)
M = len(h)
N_oa = L + M - 1 # FFT size for Overlap-Add

# Block size for Overlap-Save (L = N - M + 1)
L_os = 10
N_os = L_os + M - 1

# Perform convolution using all three methods for comparison
y_oa = overlap_add(x, h, L)
y_os = overlap_save(x, h, L_os)
y_conv = np.convolve(x, h)

# Plotting the results
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.stem(np.arange(len(x)), x)
plt.title('Input Signal x')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(np.arange(len(y_oa)), y_oa)
plt.title('Convolution using Overlap-Add Method')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.stem(np.arange(len(y_os)), y_os)
plt.title('Convolution using Overlap-Save Method')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.stem(np.arange(len(y_conv)), y_conv)
plt.title('Convolution using np.convolve (Verification)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print first few elements to verify they are close
print("First 10 values (Overlap-Add):", y_oa[:10])
print("First 10 values (Overlap-Save):", y_os[:10])
print("First 10 values (np.convolve):", y_conv[:10])