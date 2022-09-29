# FIRfilter
FIR filter scipy usage note

![lowpass](https://github.com/tk-yoshimura/FIRfilter/blob/main/figures/lowpass.svg)

## Usage

```py
from firwin import *

sample_hz = 40000
N = 8192

filter = LowPass(sample_hz, cutoff_hz=200)

x = np.random.normal(size=(2, 3, N))
y = filter(x)
```
