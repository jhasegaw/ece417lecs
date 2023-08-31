import librosa   # librosa can read & write sound files and spectrograms
import IPython # use this one to play waveforms in the Jupyter notebook
import numpy as np   # We use this one to do numerical operations
import urllib, io  # URLlib downloads things, io turns them into file-like objects
import matplotlib.pyplot as plt  # We use this one to plot things
import matplotlib.figure, matplotlib.gridspec
import argparse, os

def make_unknown_phase():
    '''
    \frametitle{Example: We don't know the phase}
    \centerline{\includegraphics{height=0.8\textheight}{exp/unknown_phase.png}}
    mag, phase, inverse for true, zero, and random phase?
    '''

    # Get a waveform, convert to frames, and window
    url = 'https://upload.wikimedia.org/wikipedia/commons/a/ac/Voiceless_alveolar_sibilant.ogg'
    wav,fs = librosa.load(io.BytesIO(urllib.request.urlopen(url).read()), sr=16000)
    frame_len = int(0.02*fs)
    hop_length = int(0.01*fs)
    windowed_frames = np.array([wav[t*hop_length:t*hop_length+frame_len] for t in range(int((len(wav)-frame_len)/hop_length)) ]) * np.hanning(frame_len)
    
    fig, ax = plt.subplots(3,3,figsize=(18,10))
    
    frame_num = int(0.4*fs/hop_length) # start 0.4s in
    freq_axis = np.arange(frame_len)*fs/frame_len
    time_axis = np.arange(frame_len)/fs
    spec = np.fft.fft(windowed_frames[frame_num,:])
    ax[0,0].plot(freq_axis,np.abs(spec))
    ax[0,0].set_title('Magnitude FFT')
    ax[1,0].plot(freq_axis,np.angle(spec))
    ax[1,0].set_title('Correct Phase')
    ax[1,0].set_xlabel('Frequency (Hz)')
    ax[2,0].plot(time_axis,windowed_frames[frame_num,:])
    ax[2,0].set_title('Time-Domain Signal')
    ax[2,0].set_xlabel('Time (s)')
    ax[0,1].plot(freq_axis,np.abs(spec))
    ax[0,1].set_title('Magnitude FFT')
    ax[1,1].plot(freq_axis,np.zeros(frame_len))
    ax[1,1].set_title('Zero Phase')
    ax[1,1].set_xlabel('Frequency (Hz)')
    ax[2,1].plot(time_axis,np.real(np.fft.ifft(np.abs(spec))))
    ax[2,1].set_title('Time-Domain Signal')
    ax[2,1].set_xlabel('Time (s)')
    ax[0,2].plot(freq_axis,np.abs(spec))
    ax[0,2].set_title('Magnitude FFT')
    r = np.random.rand(int(frame_len/2)-1)*2*np.pi
    random_phase = np.concatenate((np.zeros(1),r,np.zeros(1),r[::-1]))
    ax[1,2].plot(freq_axis,random_phase)
    ax[1,2].set_title('Random Phase')
    ax[1,2].set_xlabel('Frequency (Hz)')
    ax[2,2].plot(time_axis,np.real(np.fft.ifft(np.abs(spec)*np.exp(1j*random_phase))))
    ax[2,2].set_title('Time-Domain Signal')
    ax[2,2].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig('exp/unknown_phase.png')

def make_twoconstraints():
    '''
    \frametitle{Combining the two constraints}
    \centerline{\includegraphics{height=0.8\textheight}{exp/twoconstraints.png}}  
    Re(X[k]) and Im(X[k]) axes.  |X[k]|=M[k] constraint is a circle.
    x[n]=sum X[k]e^{j\omega n}=sum Re(X[k])cos(omega n)-Im(X[k])sin(omega n) is a plane.
    '''
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot([-1.5,1.5],[0,0],'k--',label='_')
    ax.plot([0,1e-6],[-1.5,1.5],'k--',label='_')
    theta=np.linspace(0,2*np.pi,360)
    ax.plot(np.cos(theta),np.sin(theta),'C0',label='Magnitude constraint')
    xr = np.array([-1.5,1.5])
    wts=np.array([-1.333,4])
    ax.plot(xr,(1-xr*wts[0])/wts[1],'C1',label='Linear constraint')
    ax.legend()
    ax.set_xlabel('Real Part $X_r[k]$',fontsize=18)
    ax.set_ylabel('Imaginary Part $X_i[k]$',fontsize=18)
    fig.tight_layout()
    fig.savefig('exp/twoconstraints.png')
    
    #  \frametitle{Griffin-Lim initialization: Random phase}
    #  \centerline{\includegraphics{height=0.8\textheight}{exp/twoconstraints_initialization.png}}
    # twoconstraints.png plus starting point with |X[k]|=M[k] and random phase
    x0m = np.array([np.cos(-np.pi/6),np.sin(-np.pi/6)])
    ax.scatter(x0m[0],x0m[1],s=320,c='r',marker='o')
    fig.savefig('exp/twoconstraints_initialization.png')
    
    #  \frametitle{Orthogonal projection}
    #  \centerline{\includegraphics{height=0.8\textheight}{exp/twoconstraints_projection.png}}  
    # twoconstraints.png plus projection of starting point onto the plane
    what = wts/np.linalg.norm(wts)
    x0l = np.array([what[1],-what[0]])*np.dot([what[1],-what[0]],x0m)+what/np.linalg.norm(wts)
    ax.plot([x0m[0],x0l[0]],[x0m[1],x0l[1]],'r',linewidth=7)
    ax.scatter(x0l[0],x0l[1],s=320,c='r',marker='o')
    fig.savefig('exp/twoconstraints_projection.png')
    
    #  \frametitle{Adjusting the magnitude}
    #  \centerline{\includegraphics{height=0.8\textheight}{exp/twoconstraints_magnitude.png}}  
    x1m = x0l / np.linalg.norm(x0l)
    ax.plot([x0l[0],x1m[0]],[x0l[1],x1m[1]],'r',linewidth=7)
    ax.scatter(x1m[0],x1m[1],s=320,c='r',marker='o')
    fig.savefig('exp/twoconstraints_magnitude.png')
    
    #  \frametitle{Iterate}
    #  \centerline{\includegraphics{height=0.8\textheight}{exp/twoconstraints_iterate.png}}  
    x1l =  np.array([what[1],-what[0]])*np.dot([what[1],-what[0]],x1m)+what/np.linalg.norm(wts)
    ax.plot([x1m[0],x1l[0]],[x1m[1],x1l[1]],'r',linewidth=7)
    ax.scatter(x1l[0],x1l[1],s=320,c='r',marker='o')
    x2m = x1l / np.linalg.norm(x1l)
    ax.plot([x1l[0],x2m[0]],[x1l[1],x2m[1]],'r',linewidth=7)
    ax.scatter(x2m[0],x2m[1],s=320,c='r',marker='o')
    fig.savefig('exp/twoconstraints_iterate.png')

def plot_cosine_frames(sample_axis,sig,win,spec,siglabel,hop_length):
    '''
    Top row: Magnitude FFT
    Second row: Phase FFT
    Third row: Frame 2 signal
    Fourth row: Frame 1 signal
    Fifth row: Original or reconstructed signal
    '''
    fig = matplotlib.figure.Figure(figsize=(14,14))
    gs = matplotlib.gridspec.GridSpec(5,2,figure=fig)
    ax = [fig.add_subplot(gs[4,:]), fig.add_subplot(gs[3,:]), fig.add_subplot(gs[2,:])]
    ax[0].plot(sample_axis,sig)
    ax[1].plot(sample_axis,np.concatenate((win[0,:],np.zeros(hop_length))))
    ax[2].plot(sample_axis,np.concatenate((np.zeros(hop_length),win[1,:])))
    ax[0].set_title('%s Signal $x[n]$'%(siglabel),fontsize=18)
    ax[1].set_title('%s First Window $x_0[n]$'%(siglabel),fontsize=18)
    ax[2].set_title('%s Second Window $x_1[n]$'%(siglabel),fontsize=18)
    for col in range(2):
        ax = fig.add_subplot(gs[0,col])
        ax.plot(np.abs(spec[col,:]))
        ax.set_title('$|X_%d[k]|$'%(col),fontsize=18)
        ax = fig.add_subplot(gs[1,col])
        ax.plot(np.angle(spec[col,:]))
        ax.set_title('$\\angle X_%d[k]$, %s'%(col,siglabel),fontsize=18)
    return fig, ax

def plot_griffin_sequence(sample_axis,signals,wins,ffts,nrows,speccols,arrcols):
    '''
    First column: Original signal
    Next 2 columns: arrow
    Next 2 columns: FFTs with random phase
    Next 2 columns: arrow
    Next column: reconstruction from FFT w/random phase
    Next 2 columns: arrow
    Next 2 columns: FFTs from reconstruction
    Next 2 columns: arrow
    Next 2 columns: FFTs with magnitude corrected
    '''
    fig = matplotlib.figure.Figure(figsize=(4*nrows,8))
    gs = matplotlib.gridspec.GridSpec(4,2*nrows,figure=fig)
    
    fig.add_subplot(gs[:,0]).plot(signals[0],sample_axis)
    fig.add_subplot(gs[:,7]).plot(signals[1],sample_axis)
    
    for reconnum,startcol in enumerate(speccols):
        for t in range(2):
            ax = fig.add_subplot(gs[2*t,startcol:startcol+2])
            ax.plot(np.abs(ffts[reconnum+1][t,:]))
            ax.set_title('$|X_%d[k]|$'%(t),fontsize=18)
            ax = fig.add_subplot(gs[2*t+1,startcol:startcol+2])
            ax.plot(np.angle(ffts[reconnum+1][t,:]))
            ax.set_title('$\\angle X_%d[k]$'%(t),fontsize=18)
            
    for startcol in arrcols:
        ax = fig.add_subplot(gs[:,startcol:startcol+2])
        ax.arrow(0,0,10,0,width=10)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax

def plot_iteration(sample_axis,signals,wins,ffts,specnums):
    fig = matplotlib.figure.Figure(figsize=(24,8))
    gs = matplotlib.gridspec.GridSpec(4,12,figure=fig)
        
    fig.add_subplot(gs[:,0]).plot(signals[specnums[0]],sample_axis)
    fig.add_subplot(gs[:,11]).plot(signals[specnums[1]],sample_axis)
        
    for reconnum,startcol in zip(specnums,[3,7]):
        for t in range(2):
            ax = fig.add_subplot(gs[2*t,startcol:startcol+2])
            ax.plot(np.abs(ffts[reconnum][t,:]))
            ax.set_title('$|X_%d[k]|$'%(t),fontsize=18)
            ax = fig.add_subplot(gs[2*t+1,startcol:startcol+2])
            ax.plot(np.angle(ffts[reconnum][t,:]))
            ax.set_title('$\\angle X_%d[k]$'%(t),fontsize=18)
                
    for startcol in [1,5,9]:
        ax = fig.add_subplot(gs[:,startcol:startcol+2])
        ax.arrow(0,0,10,0,width=10)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax
                
    
def make_cosine_stft(whichfig):
    '''
    ####################################################
    ####################################################
    #  \frametitle{STFT of a cosine}
    #  \centerline{\includegraphics{height=0.8\textheight}{exp/cosine_stft.png}}
    # Cosine at bottom, then two Hamming-windowed segments, then mag and phase STFT?
    '''
    frame_len = 200
    hop_length = 100
    T0 = frame_len/10.5

    ham = np.hamming(frame_len)
    
    # signals[0], wins[0], ffts[0]: Original
    sample_axis = np.arange(frame_len+hop_length)
    signals = [ np.cos(2*np.pi*sample_axis / T0) ]
    wins = [ np.array([signals[0][:frame_len],signals[0][hop_length:hop_length+frame_len]]) ]
    wins[0] *= np.hanning(frame_len)
    ffts = [ np.fft.fft(wins[0]) ]

    # ffts[1], wins[1], signals[1]: reconstructed with random phase
    rng = np.random.default_rng(seed=42)
    r = rng.random((2,int(frame_len/2)-1))*2*np.pi
    random_phase = np.hstack((np.zeros((2,1)),r,np.zeros((2,1)),r[:,::-1]))
    ffts.append(np.abs(ffts[0])*np.exp(1j*random_phase))
    wins.append(np.real(np.fft.ifft(ffts[1])))
    signals.append(np.concatenate((wins[1][0,:],np.zeros(hop_length))))
    signals[-1] += np.concatenate((np.zeros(hop_length),wins[1][1,:]))
    signals[-1][:hop_length] /= ham[:hop_length]
    signals[-1][-hop_length:] /= ham[-hop_length:]

    for iterations in range(20):
        # signals[2], wins[2], ffts[2]: computed from the reconstruction
        signals.append(signals[-1][:])
        wins.append(np.array([signals[-1][:frame_len],signals[-1][hop_length:hop_length+frame_len]]))
        wins[-1] *= np.hanning(frame_len)
        ffts.append(np.fft.fft(wins[-1]))
        
        # ffts[3], wins[3], signals[3]: normalize to the magnitude of ffts[0]
        ffts.append(np.abs(ffts[0])*np.exp(1j*np.angle(ffts[-1])))
        wins.append(np.real(np.fft.ifft(ffts[-1])))
        signals.append(np.concatenate((wins[-1][0,:],np.zeros(hop_length))))
        signals[-1] += np.concatenate((np.zeros(hop_length),wins[-1][1,:]))
        signals[-1][:hop_length] /= ham[:hop_length]
        signals[-1][-hop_length:] /= ham[-hop_length:]
    
    if whichfig=='cosine_stft':
        fig, ax = plot_cosine_frames(sample_axis,signals[0],wins[0],ffts[0],'Original',hop_length)
    elif whichfig=='cosine_mstft':
        win = np.real(np.fft.ifft(np.abs(ffts[0])))
        sig = np.concatenate((win[0,:],np.zeros(hop_length)))
        sig += np.concatenate((np.zeros(hop_length),win[1,:]))
        sig[:hop_length] /= ham[:hop_length]
        sig[-hop_length:] /= ham[-hop_length:]
        fig, ax = plot_cosine_frames(sample_axis,sig,win,np.abs(ffts[0]),'Zero-Phase',hop_length)
    elif whichfig=='cosine_rstft':
        fig, ax = plot_cosine_frames(sample_axis,signals[1],wins[1],ffts[1],'Random-Phase',hop_length)
    elif whichfig=='cosine_ola':
        fig, ax = plot_griffin_sequence(sample_axis,signals,wins,ffts,4,[3],[1,5])
    elif whichfig=='cosine_stft2':
        fig, ax = plot_griffin_sequence(sample_axis,signals,wins,ffts,6,[3,10],[1,5,8])
    elif whichfig=='cosine_mstft2':
        fig, ax = plot_griffin_sequence(sample_axis,signals,wins,ffts,8,[3,10,14],[1,5,8,12])
    elif whichfig=='cosine_iterate1':
        fig, ax = plot_iteration(sample_axis,signals,wins,ffts,[2,3])
    elif whichfig=='cosine_iterate2':
        fig, ax = plot_iteration(sample_axis,signals,wins,ffts,[4,5])
    elif whichfig=='cosine_iterate3':
        fig, ax = plot_iteration(sample_axis,signals,wins,ffts,[6,7])
    elif whichfig=='cosine_iterate4':
        fig, ax = plot_iteration(sample_axis,signals,wins,ffts,[8,-1])

    fig.tight_layout()
    fig.savefig('exp/%s.png'%(whichfig))


#################################################################
# What to do if somebody calls this from the command line:
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create figures',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('whichfig')
    args = parser.parse_args()

    os.makedirs('exp',exist_ok=True)
    
    if args.whichfig=='unknown_phase':
        make_unknown_phase()
    elif args.whichfig=='twoconstraints':
        make_twoconstraints()
    else:
        make_cosine_stft(args.whichfig)
