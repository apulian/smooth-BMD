"""
% This script performs the numerical computations described in Section IV.A
% of the following article by G. Patera and A. Pugliese (in particular, see
% Figure 2)
"""

import numpy as np
import matplotlib.pyplot as plt
from smooth_bmd import smooth_bmd

# Load data (assumes presence of 'example_smoothBMD_data.npz' with keys 'M' and 'dGAMMA')
data = np.load('example_smoothBMD_data.npz')
M = data['M']
dGAMMA = data['dGAMMA']

dGAMMA = dGAMMA.flatten()
I = np.eye(M.shape[0])
D = np.sqrt(2) * np.diag(np.sqrt(dGAMMA))

# Define the matrix-valued function
def matfun(w):
    return D @ np.linalg.inv(np.diag(dGAMMA + 1j * w) - M) @ D - I

# Compute joint-minimum-variation ABMD
tspan = [0, 5]
Tout, Uout, Dout, Vout, flag = smooth_bmd(matfun, tspan)

# Plot output
plt.figure()
for k in range(Dout.shape[0]):
    plt.plot(Tout, 20 * np.log10(Dout[k, :]), '-', linewidth=2)
plt.title('Squeezing spectrum')
plt.xlabel('Frequency')
plt.ylabel('Squeezing level (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()