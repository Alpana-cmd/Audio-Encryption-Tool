import numpy as np
def logistic_map(x, r):
    return r * x * (1 - x)

def henon_map(x, y, a, b):
    return y + 1 - a * x ** 2, b * x

# Example usage
x = 0.5
r = 3.8
y = 0.5
a, b = 1.4, 0.3
for i in range(10):
    x = logistic_map(x, r)
    y, x = henon_map(x, y, a, b)
    print(x, y)

# Define function to divide an audio signal into segments of a specified length
def divide_signal(signal, segment_length):
    num_segments = int(np.ceil(len(signal) / segment_length))
    padded_length = num_segments * segment_length
    padded_signal = np.pad(signal, (0, padded_length - len(signal)), mode='constant')
    segments = np.reshape(padded_signal, (num_segments, segment_length))
    return segments

# Example usage
audio_signal = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
segment_length = 4
segments = divide_signal(audio_signal, segment_length)
print(segments)

# Define chaotic maps
def map1(x, a=3.8):
    return a * x * (1 - x)

def map2(x, a=3.7):
    return a * x * (1 - x)

# Define function to generate key sequence using chaotic map
def generate_key_sequence(map_func, seed, length):
    x = seed
    key_sequence = np.zeros(length, dtype=int)
    for i in range(length):
        x = map_func(x)
        key_sequence[i] = int(x >= 0.5)
    return key_sequence

# Define function to generate final key sequence using two chaotic maps and XOR operation
def generate_final_key_sequence(signal, map1_seed, map2_seed, key_length):
    num_segments = len(signal) // key_length
    final_key_sequence = np.zeros(num_segments * key_length, dtype=int)
    for i in range(num_segments):
        segment = signal[i*key_length:(i+1)*key_length]
        key1 = generate_key_sequence(map1, map1_seed, key_length)
        key2 = generate_key_sequence(map2, map2_seed, key_length)
        final_key_sequence[i*key_length:(i+1)*key_length] = np.bitwise_xor(segment, np.bitwise_xor(key1, key2))
    return final_key_sequence

# Example usage
signal = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
map1_seed = 0.123
map2_seed = 0.456
key_length = 4
final_key_sequence = generate_final_key_sequence(signal, map1_seed, map2_seed, key_length)
print(final_key_sequence)


# Define DNA encoding scheme
DNA_ENCODING = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T'
}

# Define function to perform dynamic diffusion on a segment of audio signal using key sequence
def dynamic_diffusion(segment, key_sequence):
    # Permutation operation
    permuted_segment = np.random.permutation(segment)

    # Substitution operation
    substituted_segment = np.zeros_like(segment)
    for i, sample in enumerate(permuted_segment):
        substituted_segment[i] = (sample + key_sequence[i]) % 4

    return substituted_segment

# Define function to encode a segment of audio signal as a DNA sequence
def dna_encode(segment):
    dna_sequence = ''
    for sample in segment:
        dna_sequence += DNA_ENCODING[sample]
    return dna_sequence

# Example usage
segment = np.array([1, 0, 0, 1])
key_sequence = np.array([1, 0, 1, 0])
encrypted_segment = dynamic_diffusion(segment, key_sequence)
dna_sequence = dna_encode(encrypted_segment)
print(dna_sequence)


# Define function to combine multiple encrypted audio segments into a single encrypted audio signal
def combine_segments(segments):
    # Calculate the total length of the encrypted audio signal
    signal_length = sum([len(segment) for segment in segments])

    # Create an array to hold the encrypted audio signal
    encrypted_signal = np.zeros(signal_length, dtype=int)

    # Copy the encrypted segments into the encrypted audio signal array
    index = 0
    for segment in segments:
        encrypted_signal[index:index+len(segment)] = segment
        index += len(segment)

    return encrypted_signal

# Example usage
segment1 = np.array([1, 2, 0, 3])
segment2 = np.array([0, 3, 2])
segment3 = np.array([2, 1])
encrypted_segment1 = np.array([3, 1, 2, 0])
encrypted_segment2 = np.array([2, 1, 3])
encrypted_segment3 = np.array([1, 3])
encrypted_signal = combine_segments([encrypted_segment1, encrypted_segment2, encrypted_segment3])
print(encrypted_signal)

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
from numpy import savetxt


# Define chaotic maps
def logistic_map(x, r):
    return r * x * (1 - x)

def henon_map(x, y, a, b):
    return y + 1 - a * x ** 2, b * x

# Segment audio signal
def segment_signal(signal, segment_length):
    num_samples = len(signal)
    num_segments = num_samples // segment_length
    segments = np.zeros((num_segments, segment_length))
    for i in range(num_segments):
        segments[i] = signal[i * segment_length:(i + 1) * segment_length]
    return segments

# Load audio file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.flac;*.mp3")])
signal, sr = sf.read(file_path)

# Set segmentation parameters
segment_length = 2048
num_segments = len(signal) // segment_length

# Initialize chaos maps
x_lsc, x_hn, y_hn = 0.5, 0.5, 0.5
r_lsc = 3.8
a_hn, b_hn = 1.4, 0.3

# Generate encryption key
np.random.seed(1234)
key = np.random.randint(0, 256, size=signal.shape[0])
key = np.reshape(key, (-1, 2))
key = np.random.randint(0, 256, size=signal.shape).astype(np.uint8)

# Encrypt audio signal
encrypted_signal = signal.astype(np.uint8) ^ key

# Convert encrypted signal to float32 datatype
encrypted_signal = encrypted_signal.astype('float32')

# Initialize output array
output = np.zeros_like(encrypted_signal)

# Process each segment
for i in range(num_segments):
    # Apply LSC map
    for j in range(segment_length):
        x_lsc = np.cos(np.pi * (4 * r_lsc * x_lsc * (1 - x_lsc) + (1 - r_lsc) * np.sin(np.pi * x_lsc) - 0.5))

    # Apply Henon map and diffusion
    segment = encrypted_signal[i * segment_length:(i + 1) * segment_length]
    for j in range(segment_length):
        y_hn, x_hn = henon_map(x_hn, y_hn, a_hn, b_hn)
        segment[j] = segment[j] * y_hn

    # Store processed segment in output array
    output[i * segment_length:(i + 1) * segment_length] = segment

# Check if output signal is different from input signal
if np.array_equal(signal, output):
    print("Error: input and output signals are the same")

# Save encrypted audio file
output_file_path = filedialog.asksaveasfilename(defaultextension=".wav")
sf.write(output_file_path, output, sr)

# Convert output signal to int16 datatype
output_clipped = np.clip(output, -1.0, 1.0)
output_int16 = (output_clipped * np.iinfo(np.int16).max).astype(np.int16)

# Write output signal to file
sf.write(output_file_path, output_int16, sr)

# Encrypt audio file and indicate success
print("Audio file successfully encrypted!")

# Encrypt audio file and write flag to separate file
flag_file = open("encrypted.flag", "w")
flag_file.write("This file has been encrypted.")
flag_file.close()
# Save encryption key to file
key_file_path = filedialog.asksaveasfilename(defaultextension=".txt")
with open(key_file_path, 'w') as f:
    for row in key:
        np.savetxt(f, row, fmt='%d')
print("Encryption key saved to file:", key_file_path)

import wave

# Open the audio file in read-only mode
with wave.open(r"C:\Users\Hp\OneDrive\Desktop\My audio wav\1f7f8b00-cae4-11ed-ad44-db82a9e50521.wav", 'rb') as audio_file:
    # Get the audio file's parameters
    params = audio_file.getparams()

    # Read the audio frames and convert them to binary data
    audio_frames = audio_file.readframes(params.nframes)

# Print the audio file's parameters
print(params)

# Print the number of audio frames
print(len(audio_frames))

import numpy as np

# Define the Sine-Cosine map permutation function
def sine_cosine_map(x, a=0.5, mu=3.8):
    return a * np.sin(mu * np.pi * x) + (1 - a) * np.cos(mu * np.pi * x)

# Load the audio file and convert it to a numpy array
with wave.open(r"C:\Users\Hp\OneDrive\Desktop\My audio wav\1f7f8b00-cae4-11ed-ad44-db82a9e50521.wav", 'rb') as audio_file:
    params = audio_file.getparams()
    audio_frames = audio_file.readframes(params.nframes)
    audio_samples = np.frombuffer(audio_frames, dtype=np.int16)

# Define the initial value for the Sine-Cosine map
x0 = 0.1

# Generate the permutation sequence using the Sine-Cosine map
permutation_sequence = [int(np.abs(sine_cosine_map(x0 + i))) % len(audio_samples) for i in range(len(audio_samples))]

# Permute the audio samples using the permutation sequence
permuted_samples = audio_samples[permutation_sequence]

# Convert the permuted samples back to binary data
permuted_frames = permuted_samples.tobytes()

# Write the permuted audio data to a new file
with wave.open('permuted_audio.wav', 'wb') as permuted_file:
    permuted_file.setparams(params)
    permuted_file.writeframes(permuted_frames)

import numpy as np
import matplotlib.pyplot as plt

# Define LSC map equation
def LSC_map(x, r):
    return np.cos(np.pi * (4 * r * x * (1 - x) + (1 - r) * np.sin(np.pi * x) - 0.5))

# Define initial seed and range of parameters
x0 = 0.1
r_values = np.linspace(0, 1, 100000)

# Compute LSC map iteratively for each parameter value
iterations = 100
x_values = []
y_values = []
for r in r_values:
    x = x0
    for i in range(iterations):
        x = LSC_map(x, r)
    x_values.append(r)
    y_values.append(x)

# Plot the LSC Chaotic Map as a scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_values, y_values, s=1, c='red') # make the dots red
ax.set_xlabel('r')
ax.set_ylabel('x')
ax.set_title('Bifurcation of Logistic Chaotic Map')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the Henon map equations
def henon_map(x, y, a, b):
    return y + 1 - a * x**2, b * x

# Define the number of iterations and discard iterations
num_iter = 100
discard_iter = 50

# Define the range of parameter values
a_values = np.linspace(1, 1.4, 1000)
b_values = np.linspace(0, 0.3, 1000)

# Initialize an empty array to store the values of x
x_values = np.empty((len(a_values), num_iter - discard_iter))

# Iterate over each value of a and b
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        # Initialize the values of x and y
        x = 0.1
        y = 0.1
        # Iterate over the specified number of iterations
        for k in range(num_iter):
            # Discard the first few iterations to allow the system to settle
            if k >= discard_iter:
                x_values[i, k - discard_iter] = x
            # Apply the Henon map equations
            x, y = henon_map(x, y, a, b)

# Plot the bifurcation diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(np.tile(a_values, num_iter - discard_iter), x_values.flatten(), s=0.1, c='black', alpha=0.5)
ax.set_xlim([1, 1.4])
ax.set_ylim([-0.5, 0.5])
ax.set_xlabel('a')
ax.set_ylabel('x')
ax.set_title('Bifurcation Diagram for the Henon Map')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def lsc_map(x, r):
    return np.cos(np.pi * (4 * r * x * (1 - x) + (1 - r) * np.sin(np.pi * x) - 0.5))

def lyapunov_exponent_lsc(r, x0, n=1000):
    x = x0
    lyap = 0
    for i in range(n):
        fx = lsc_map(x, r)
        dfx = -np.pi * (4 * r * (1 - 2 * x) + (1 - r) * np.cos(np.pi * x))
        lyap += np.log(abs(dfx))
        x = fx
    return lyap / n

r_values = np.linspace(0, 8, 4000)
lyapunov_exponents = [lyapunov_exponent_lsc(r, 0.1) for r in r_values]

plt.plot(r_values, lyapunov_exponents)
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of Logistic Sine Cosine map')
plt.xticks(np.arange(0, 9, 2))
plt.yticks(np.arange(-3, 3, 1))
plt.ylim(-3, 2)
plt.show()

