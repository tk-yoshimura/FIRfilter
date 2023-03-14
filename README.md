# FIR filter
FIR(Finite Impulse Response) filter scipy usage note

![lowpass](https://github.com/tk-yoshimura/FIRfilter/blob/main/figures/lowpass.svg)

## Usage

```py
from firwin import *

batches, channels, length = 2, 3, 8192
filter = LowPass(sample_hz=40000, cutoff_hz=200)

x = np.random.normal(size=(batches, channels, length))
y = filter(x)
```
