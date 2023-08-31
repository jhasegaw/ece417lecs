import numpy as np   # We use this one to do numerical operations
import urllib, io  # URLlib downloads things, io turns them into file-like objects
import matplotlib.pyplot as plt  # We use this one to plot things

w = np.hanning(200)
fig, ax = plt.subplots(1,1,figsize=(14,1))
ax.plot(w)
fig.savefig('exp/hamming_window.png')

