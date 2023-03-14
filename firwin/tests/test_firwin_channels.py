import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from firwin import FIRChannels

sample_hz = 16384
N = 8192

filter = FIRChannels(sample_hz)

assert filter(np.zeros(256)).shape == (8, 256), 'incorrect reshape dim1'
assert filter(np.zeros((3, 256))).shape == (3, 8, 256), 'incorrect reshape dim2'
assert filter(np.zeros((7, 3, 256))).shape == (7, 3, 8, 256), 'incorrect reshape dim3'

ws = np.sum(filter.weight, axis=0)

plt.clf()
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(ws)) - len(ws)//2, np.abs(ws))
plt.yscale('log')
plt.show()

t = np.arange(N) / sample_hz

x = (np.sin(t * 100) + np.sin(t * 200 + 1) + np.sin(t * 400 + 2) + np.sin(t * 800 + 3) + 
     np.sin(t * 1600 + 4) / 2 + np.sin(t * 3200 + 5) / 4 + np.sin(t * 6400 + 6) / 8 + np.sin(t * 12800 + 6) / 16)
y = filter(x)

plt.clf()
plt.figure(figsize=(12, 5))
plt.plot(t, x, label='raw')

for hz, c in zip(filter.cutoff_hz, y):
    plt.plot(t, c, label='fir filtered %.1f-%.1f' % (hz[0], hz[1]))

plt.plot(t, np.sum(y, axis = 0), label='sum', linestyle='dashed', linewidth=1, color='black')

plt.xlim([0, 0.1])

plt.legend(loc='lower right')
plt.grid(which='both', axis='both')
plt.show()

x = (np.sin(t * 1600 + 4) / 2 + np.sin(t * 3200 + 5) / 4 + np.sin(t * 6400 + 6) / 8 + np.sin(t * 12800 + 6) / 16)
y = filter(x)

plt.clf()
plt.figure(figsize=(12, 5))
plt.plot(t, x, label='raw')

for hz, c in zip(filter.cutoff_hz, y):
    plt.plot(t, c, label='fir filtered %.1f-%.1f' % (hz[0], hz[1]))

plt.plot(t, np.sum(y, axis = 0), label='sum', linestyle='dashed', linewidth=1, color='black')

plt.xlim([0, 0.1])

plt.legend(loc='lower right')
plt.grid(which='both', axis='both')
plt.show()

x = (np.sin(t * 16384 + 6) / 16)
y = filter(x)

plt.clf()
plt.figure(figsize=(12, 5))
plt.plot(t, x, label='raw')

for hz, c in zip(filter.cutoff_hz, y):
    plt.plot(t, c, label='fir filtered %.1f-%.1f' % (hz[0], hz[1]))

plt.plot(t, np.sum(y, axis = 0), label='sum', linestyle='dashed', linewidth=1, color='black')

plt.xlim([0, 0.1])

plt.legend(loc='lower right')
plt.grid(which='both', axis='both')
plt.show()