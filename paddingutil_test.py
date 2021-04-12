import numpy as np
from paddingutil import padding_edge_reflect
import matplotlib.pyplot as plt

x = np.arange(16) * 0.2 + 4  
#x = np.random.normal(size = (16))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15))
plt.show()

x = x.reshape((16, 1))
#x = np.random.normal(size = (16, 1))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15, 0)[:, 0])
plt.show()

x = x.reshape((1, 16))
#x = np.random.normal(size = (1, 16))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15, 1)[0])
plt.show()

x = x.reshape((16, 1, 1))
#x = np.random.normal(size = (16, 1, 1))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15, 0)[:, 0, 0])
plt.show()

x = x.reshape((1, 16, 1))
#x = np.random.normal(size = (1, 16, 1))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15, 1)[0, :, 0])
plt.show()

x = x.reshape((1, 1, 16))
#x = np.random.normal(size = (1, 1, 16))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(46), padding_edge_reflect(x, 15, 2)[0, 0, :])
plt.show()

x = np.random.normal(size = (16, 2, 3))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(24), padding_edge_reflect(x, 4, 0)[:, 0, 0])
plt.show()

x = np.random.normal(size = (3, 16, 4))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(24), padding_edge_reflect(x, 4, 1)[0, :, 0])
plt.show()

x = np.random.normal(size = (5, 6, 16))

plt.figure(figsize=(12, 5))
plt.plot(np.arange(24), padding_edge_reflect(x, 4, 2)[0, 0, :])
plt.show()