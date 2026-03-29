#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Read the binary file directly into a numpy array of 64-bit floats
data = np.fromfile('freq.bin', dtype=np.float64)

# Plot the array
plt.plot(data)
plt.title("DFT Magnitude of 440Hz Sine Wave")
plt.xlabel("Frequency Bin (k)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
