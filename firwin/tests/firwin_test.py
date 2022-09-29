import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from firwin import *

sample_hz = 40000
N = 8192

filters = [    
    LowPass(sample_hz, cutoff_hz=200),
    HighPass(sample_hz, cutoff_hz=200),
    BandPass(sample_hz, lower_cutoff_hz=300, higher_cutoff_hz=500),
    BandCut(sample_hz, lower_cutoff_hz=300, higher_cutoff_hz=500),
]

for filter in filters:
    w, h = signal.freqz(filter.firwin)
 
    a = 20 * np.log10(abs(h));
    f = w / (2 * np.pi) * sample_hz

    plt.figure(figsize=(12, 5))
    plt.xscale("log")
    plt.plot(f, a)
    plt.xlabel('freq. [Hz]')
    plt.ylabel('amp. [dB]')
    plt.margins(0, 0.1)
    plt.xlim(10, sample_hz)
    plt.ylim([-8, 2])
    plt.grid(which='both', axis='both')
    plt.show()

    freqs = np.fft.fftfreq(N, 1 / sample_hz)
    fn = len(freqs) // 2

    x = np.random.normal(size=N)
    y = filter(x)
    y_fft = np.fft.fft(y)
    y_psd = np.sqrt(np.array([c.real * c.real + c.imag * c.imag for c in y_fft]))

    plt.figure(figsize=(12, 5))
    plt.xscale("log")
    plt.plot(freqs[:fn], y_psd[:fn])
    plt.xlabel('freq. [Hz]')
    plt.ylabel('PSD')
    plt.xlim(10, sample_hz)
    plt.grid(which='both', axis='both')
    plt.show()

    t = np.arange(N) / sample_hz
    x = np.sin(t * 10) + np.sin(t * 20 + 1) + np.sin(t * 40 + 2) + np.sin(t * 80 + 3) + np.sin(t * 160 + 4)
    x += np.sin(t * 320 + 5) + np.sin(t * 640 + 6) + np.sin(t * 1280 + 7) + np.sin(t * 2560 + 8)
    y = filter(x)

    plt.figure(figsize=(12, 5))
    plt.plot(t, x, label='raw')
    plt.plot(t, y, label='fir filtered')
    plt.xlabel('time')
    plt.legend()
    plt.grid(which='both', axis='both')
    plt.show()