import numpy as np
from firwin import pad_boundary
import matplotlib.pyplot as plt

x = np.random.normal(size=(16))
y = pad_boundary(x, 15)

assert y.shape == (16 + 15 * 2,)

x = np.random.normal(size=(5, 16))
y = pad_boundary(x, 15)

assert y.shape == (5, 16 + 15 * 2,)

x = np.random.normal(size=(2, 5, 16))
y = pad_boundary(x, 15)

assert y.shape == (2, 5, 16 + 15 * 2,)

def plot(x, padwidth):
    y = pad_boundary(x, padwidth)

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(x.shape[-1] + padwidth * 2), y, linestyle="dashed")
    plt.plot(np.arange(x.shape[-1]) + padwidth, x)
    plt.show()

x = np.random.normal(size=(16))

plot(x, 15)

x = np.random.normal(size=(128))

plot(x, 15)
plot(x, 25)

x = np.random.normal(size=(16)) + np.sin(np.linspace(0, 6, 16)) * 4

plot(x, 15)

x = np.random.normal(size=(128)) + np.sin(np.linspace(0, 6, 128)) * 4

plot(x, 15)
plot(x, 25)

x = np.random.normal(size=(128)) + (np.sin(np.linspace(0, 6, 128)) + np.sin(np.linspace(0, 32, 128))) * 4

plot(x, 15)
plot(x, 25)

x = np.random.normal(size=(128)) + (np.cos(np.linspace(0, 6, 128)) + np.cos(np.linspace(0, 32, 128))) * 4

plot(x, 15)
plot(x, 25)
