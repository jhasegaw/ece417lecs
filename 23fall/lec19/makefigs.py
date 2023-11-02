import librosa, os
import numpy as np
import matplotlib.pyplot as plt


os.makedirs('exp',exist_ok=True)

#######################3
# Mel filterbanks
n_mels = 40
n_freqs = 1025
sr = 16000
melfb = librosa.filters.mel(sr=sr, n_fft=(n_freqs-1)*2, n_mels=n_mels, norm=None)

f = np.linspace(0,int(sr/2),n_freqs)
fig, ax = plt.subplots(1,1)
ax.plot(f,melfb.T)
ax.set_xticks([0,2000,4000,6000,8000])
ax.set_xticklabels([0,2000,4000,6000,8000],fontsize=14)
ax.set_yticks([0,0.5,1])
ax.set_yticklabels([0,0.5,1],fontsize=14)
ax.set_title('Mel Filterbanks $H_m[f]$',fontsize=14)
ax.set_xlabel('Frequency $f$ (Hz)',fontsize=14)
legends = [ 'm=%d'%(m) for m in range(1,n_mels+1) ]
ax.legend(legends,fontsize=14)
fig.savefig('exp/melfilters.png')
