import numpy as np
import matplotlib.figure

#Create figures for lec03

######### Generally maybe: 1 kHz tone/filter, at 8kHz sampling rate?

fs = 8000
cf = 1000
duration = 0.02
windowlen = int(duration*fs)
hearing_threshold = 6 #dB
t = np.linspace(0,duration,windowlen,endpoint=False)
f = np.linspace(0,fs/2,int(windowlen/2+1),endpoint=True)
tone = np.sin(2*np.pi*cf*t)/(windowlen/2) * np.power(10,-hearing_threshold/20)
tone_fft = np.square(np.abs(np.fft.rfft(tone)))
white = np.random.randn(windowlen)/np.sqrt(windowlen)
white_fft = np.square(np.abs(np.fft.rfft(white)))

def hz2erbs(f):
    return(11.17268*np.log(1+46.06538*f/(f+14678.49)))
def erbs2hz(erbs):
    return(676170.4/(47.06538-np.exp(0.08950404*erbs))-14678.49)
def erb_at_hz(f):
    return(6.23*np.square(f/1000) + 93.39*(f/1000) + 28.52)
def hz2mel(f):
    return(2595*np.log(1+(f/700)))
def mel2hz(mel):
    return(700*(np.exp(mel/2595)-1))

# Create the bandstop noise by just zeroing out FFT within +/- 0.5 ERBs of cf
bandstop_fft = np.fft.rfft(white)
zero_range = (windowlen/fs) * erbs2hz(hz2erbs(cf)+np.array([-0.5,0.5]))
print('Bandstop: zeroing bins %d to %d of %d'%(int(zero_range[0]),int(zero_range[1]),int(windowlen/2+1)))
bandstop_fft[int(zero_range[0]):int(zero_range[1]+1)] = 0
bandstop  = np.fft.irfft(bandstop_fft)
bandstop_fft  = np.square(np.abs(bandstop_fft))

# tone, noise, combo plots
XY = [(t,tone,white,'waveform','white',tone+white,'time (sec)'),
      (f,tone_fft,white_fft,'powerspectrum','white',tone_fft+white_fft,'frequency (Hz)'),
      (t,tone,bandstop,'waveform','bandstop',tone+bandstop,'time (sec)'),
      (f,tone_fft,bandstop_fft,'powerspectrum','bandstop',tone_fft+bandstop_fft,'frequency (Hz)')
      ]
for xy in XY:
    fig = matplotlib.figure.Figure(figsize=(8, 4))
    axs = fig.subplots(3,1,sharex=True)
    axs[0].plot(xy[0],xy[1])
    axs[0].set_title('1kHz tone %s'%(xy[3]))
    if xy[3]=='waveform':
        axs[1].plot(xy[0],xy[2])
        axs[2].plot(xy[0],xy[5])
    else:
        axs[1].plot(xy[0],xy[2],'-',xy[0],[ 1 for f in xy[0] ],'--')
        axs[2].plot(xy[0],xy[5],'-',xy[0],[ 1 for f in xy[0] ],'--')
    axs[1].set_title('%s noise %s'%(xy[4], xy[3]))
    axs[2].set_title('tone + %s noise %s'%(xy[4],xy[3]))
    axs[2].set_xlabel('%s'%(xy[6]))
    fig.savefig('exp/tone_%s_%s.png'%(xy[4],xy[3]))

# Create and plot a bank of gammatone filters
gtfilters = np.zeros((26,windowlen))
gtfilters_fft = np.zeros((26,int(windowlen/2+1)))
for k in range(26):
    fk = erbs2hz(k+1)
    b = erb_at_hz(fk)
    gtfilters[k,:] = np.power(t*fs,3) * np.exp(-2*np.pi*b*t)*np.cos(2*np.pi*fk*t)
    gtfilters_fft[k,:] = np.square(np.abs(np.fft.rfft(gtfilters[k,:])))
    gtfilters_fft[k,:] = gtfilters_fft[k,:]/max(gtfilters_fft[k,:])
fig = matplotlib.figure.Figure(figsize=(8, 4))
ax = fig.subplots()
ax.plot(f,gtfilters_fft.T)
ax.set_title('Power spectra of auditory filters spaced 1 ERB apart')
ax.set_ylabel('Magnitude FFT')
ax.set_xlabel('Frequency (Hz)')
fig.savefig('exp/gammatone_filterbank.png')

# Create a 1kHz-centered gtfilter
b = erb_at_hz(cf)
gtfilter = np.power(t*fs,3) * np.exp(-2*np.pi*b*t)*np.cos(2*np.pi*cf*t)
gtfilter /= np.linalg.norm(gtfilter)
gtfilter /= np.sqrt(30)
gtfilter_fft = np.square(np.abs(np.fft.rfft(gtfilter)))
gt_white = np.real(np.fft.irfft(np.fft.rfft(white)*np.fft.rfft(gtfilter)))
gt_white_fft = np.square(np.abs(np.fft.rfft(gt_white)))
gt_bandstop = np.real(np.fft.irfft(np.fft.rfft(gtfilter)*np.fft.rfft(bandstop)))
gt_bandstop_fft = np.square(np.abs(np.fft.rfft(gt_bandstop)))

XY = [(t,'waveform',white,'white',gt_white,'time (sec)',gtfilter,tone),
      (f,'powerspectrum',white_fft,'white',gt_white_fft,'frequency (Hz)',gtfilter_fft,tone_fft),
      (t,'waveform',bandstop,'bandstop',gt_bandstop,'time (sec)',gtfilter,tone),
      (f,'powerspectrum',bandstop_fft,'bandstop',gt_bandstop_fft,'frequency (Hz)',gtfilter_fft,tone_fft)
      ]
for xy in XY:
    fig = matplotlib.figure.Figure(figsize=(8, 4))
    axs = fig.subplots(3,1,sharex=True)
    axs[1].plot(xy[0],xy[6])
    if xy[1]=='waveform':
        axs[0].plot(xy[0],xy[2])
        axs[1].set_title('impulse response: auditory filter centered at 1kHz')
        axs[2].plot(xy[0],xy[4])
    else:
        axs[0].plot(xy[0],xy[2],'-',xy[0],[ 1 for f in xy[0] ],'--')
        axs[1].set_title('squared frequency response: auditory filter @ 1kHz')
        axs[2].plot(xy[0],xy[4],'-',xy[0],[ 1 for f in xy[0] ],'--')
    axs[0].set_title('%s noise %s'%(xy[3],xy[1]))
    axs[2].set_title('auditory-filtered %s noise %s'%(xy[3],xy[1]))
    axs[2].set_xlabel(xy[5])
    fig.savefig('exp/gtfiltered_%s_%s.png'%(xy[3],xy[1]))
    
    fig = matplotlib.figure.Figure(figsize=(8, 4))
    axs = fig.subplots(3,1,sharex=True)
    axs[0].plot(xy[0],xy[7])
    axs[0].set_title('tone %s'%(xy[1]))
    if xy[1]=='waveform':
        axs[1].plot(xy[0],xy[4])
        axs[2].plot(xy[0],xy[4]+xy[7])
    else:
        axs[1].plot(xy[0],xy[4],'-',xy[0],[ 1 for f in xy[0] ],'--')
        axs[2].plot(xy[0],xy[4]+xy[7],'-',xy[0],[ 1 for f in xy[0] ],'--')
    axs[1].set_title('auditory-filtered %s noise %s'%(xy[3],xy[1]))
    axs[2].set_title('auditory-filtered tone + %s noise %s'%(xy[3],xy[1]))
    axs[2].set_xlabel(xy[5])
    fig.savefig('exp/gtfiltered_tone_%s_%s.png'%(xy[3],xy[1]))
    
