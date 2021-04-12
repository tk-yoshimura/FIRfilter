import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from firwin import *

sample_hz = 40000
N = 8192

filter = LowPass(sample_hz, cutoff_hz=2000)

freqs = np.fft.fftfreq(N, 1 / sample_hz)
fn = len(freqs) // 2

x = np.random.normal(size=(N, 2))
y = filter(x, axis=0)
y_fft = np.fft.fft(y[:, 1])
y_psd = np.sqrt(np.array([c.real * c.real + c.imag * c.imag for c in y_fft]))

plt.figure(figsize=(12, 5))
plt.xscale("log")
plt.plot(freqs[:fn], y_psd[:fn])
plt.xlabel('freq. [Hz]')
plt.ylabel('PSD')
plt.xlim(10, sample_hz)
plt.grid(which='both', axis='both')
plt.show()


x = np.random.normal(size=(2, N))
y = filter(x, axis=1)
y_fft = np.fft.fft(y[1, :])
y_psd = np.sqrt(np.array([c.real * c.real + c.imag * c.imag for c in y_fft]))

plt.figure(figsize=(12, 5))
plt.xscale("log")
plt.plot(freqs[:fn], y_psd[:fn])
plt.xlabel('freq. [Hz]')
plt.ylabel('PSD')
plt.xlim(10, sample_hz)
plt.grid(which='both', axis='both')
plt.show()

x = np.random.normal(size=(2, N, 3))
y = filter(x, axis=1)
y_fft = np.fft.fft(y[1, :, 1])
y_psd = np.sqrt(np.array([c.real * c.real + c.imag * c.imag for c in y_fft]))

plt.figure(figsize=(12, 5))
plt.xscale("log")
plt.plot(freqs[:fn], y_psd[:fn])
plt.xlabel('freq. [Hz]')
plt.ylabel('PSD')
plt.xlim(10, sample_hz)
plt.grid(which='both', axis='both')
plt.show()
plt.clf()