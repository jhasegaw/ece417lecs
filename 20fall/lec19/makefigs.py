#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import os,re

os.makedirs('exp',exist_ok=True)

######################################################################################3
# The input
x = np.random.randint(low=1,high=5,size=(30))
x[9]=0
x[19]=0
x[29]=0
T = len(x)
# The target
y = np.zeros(30)
y[9]=np.sum(x[0:9])
y[19]=np.sum(x[10:19])
y[29] = np.sum(x[20:29])
# Show the target
fignum = 0
f = plt.figure(fignum,figsize=(6.4,4.8))
aa = f.subplots(nrows=2,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_title('Summation RNN: Sample Input $x[t]$')
aa[1].stem(y,use_line_collection=True)
aa[1].set_title('Summation RNN: Corresponding Target Output $y[t]$')
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# Vanilla RNN, w=1
# h[t] = ReLU(x[t] + wh[t-1])
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_ylabel('x[t]')
w = 1
aa[0].set_title('Input, Target, and Output of an RNN with w=%g'%(w))
aa[1].stem(y,use_line_collection=True)
aa[1].set_ylabel('y[t]')
h = x.copy()
for t in range(1,len(h)):
    h[t] += w*h[t-1]
aa[2].stem(h,use_line_collection=True)
aa[2].set_ylabel('h[t]')
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# Vanilla RNN, w=0.5
# h[t] = ReLU(x[t] + wh[t-1])
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_ylabel('x[t]')
w = 0.5
aa[0].set_title('Input, Target, and Output of an RNN with w=%g'%(w))
aa[1].stem(y,use_line_collection=True)
aa[1].set_ylabel('y[t]')
h = x.copy()
for t in range(1,len(h)):
    h[t] += w*h[t-1]
aa[2].stem(h,use_line_collection=True)
aa[2].set_ylabel('h[t]')
f.savefig('exp/fig%d.png'%(fignum))

# Backprop
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
epsilon = np.array([ h[t]-y[t] if y[t]>0 else 0 for t in range(T) ])
aa[0].stem(epsilon,use_line_collection=True)
aa[0].set_ylabel('$\epsilon[t]=h[t]-y[t]$')
aa[0].set_title('Error $\epsilon[t]$, Defined to be nonzero only when $y[t]>0$')
delta = epsilon.copy()
for t in range(T-2,-1,-1):
    delta[t] += w*delta[t+1]
aa[1].stem(delta,use_line_collection=True)
aa[1].set_ylabel('$\delta[t]=dE/dc[t]$')
aa[1].set_title('Backprop Gradient $\delta[t]=dE/dc[t]$')
grad = np.array([0]+[delta[t]*h[t-1] for t in range(1,T)])
aa[2].stem(grad,use_line_collection=True)
aa[2].set_ylabel('$\delta[t]h[t-1]$')
aa[2].set_xlabel('$dE/dw=\sum\delta[t]h[t-1]=$%g'%(np.sum(grad)))
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# Vanilla RNN, w=1.7
# h[t] = ReLU(x[t] + wh[t-1])
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_ylabel('x[t]')
w = 1.7
aa[0].set_title('Input, Target, and Output of an RNN with w=%g'%(w))
aa[1].stem(y,use_line_collection=True)
aa[1].set_ylabel('y[t]')
h = x.copy()
for t in range(1,len(h)):
    h[t] += w*h[t-1]
aa[2].stem(h,use_line_collection=True)
aa[2].set_ylabel('h[t]')
f.savefig('exp/fig%d.png'%(fignum))

# Backprop
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
epsilon = np.array([ h[t]-y[t] if y[t]>0 else 0 for t in range(T) ])
aa[0].stem(epsilon,use_line_collection=True)
aa[0].set_ylabel('$\epsilon[t]=h[t]-y[t]$')
aa[0].set_title('Error $\epsilon[t]$, Defined to be nonzero only when $y[t]>0$')
delta = epsilon.copy()
for t in range(T-2,-1,-1):
    delta[t] += w*delta[t+1]
aa[1].stem(delta,use_line_collection=True)
aa[1].set_ylabel('$\delta[t]=dE/dc[t]$')
aa[1].set_title('Backprop Gradient $\delta[t]=dE/dc[t]$')
grad = np.array([0]+[delta[t]*h[t-1] for t in range(1,T)])
aa[2].stem(grad,use_line_collection=True)
aa[2].set_ylabel('$\delta[t]h[t-1]$')
aa[2].set_xlabel('$dE/dw=\sum\delta[t]h[t-1]=$%g'%(np.sum(grad)))
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# Ideal Forget gate
# If we could reset the _next_ sample given a zero-valued input
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_ylabel('x[t]')
w = 1.0
aa[0].set_title('Input, Target, & Output: RNN w/ Forget Gate f=CReLU(1-x[t])')
aa[1].stem(y,use_line_collection=True)
aa[1].set_ylabel('y[t]')
h = np.concatenate((np.cumsum(x[0:10]),np.cumsum(x[10:20]),np.cumsum(x[20:30])))
aa[2].stem(h,use_line_collection=True)
aa[2].set_title('Here is an $h[t]$ with zero error at the times $y[t]>0$')
aa[2].set_ylabel('h[t]')
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# Forget gate
# h[t] = ReLU(x[t] + f[t]h[t-1])
# f[t] = CReLU(x[t])
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_ylabel('x[t]')
w = 1.0
aa[0].set_title('Input, Target, & Output: RNN w/ Forget Gate f=CReLU(1-x[t])')
aa[1].stem(y,use_line_collection=True)
aa[1].set_ylabel('y[t]')
h = np.concatenate((np.cumsum(x[0:9]),np.cumsum(x[9:19]),np.cumsum(x[19:29]),[0]))
aa[2].stem(h,use_line_collection=True)
aa[2].set_ylabel('h[t]')
f.savefig('exp/fig%d.png'%(fignum))

# Backprop
fignum += 1
f = plt.figure(fignum,figsize=(6.4,6))
aa = f.subplots(nrows=3,ncols=1,sharex=True)
epsilon = np.array([ h[t]-y[t] if y[t]>0 else 0 for t in range(T) ])
aa[0].stem(epsilon,use_line_collection=True)
aa[0].set_ylabel('$\epsilon[t]=h[t]-y[t]$')
aa[0].set_title('Error $\epsilon[t]$, Defined to be nonzero only when $y[t]>0$')
delta = epsilon
for t in range(T-2,-1,-1):
    delta[t] += w*delta[t+1]
aa[1].stem(delta,use_line_collection=True)
aa[1].set_ylabel('$\delta[t]=dE/dc[t]$')
aa[1].set_title('Backprop Gradient $\delta[t]=dE/dc[t]$')
grad = np.array([0]+[delta[t]*h[t-1] for t in range(1,T)])
aa[2].stem(grad,use_line_collection=True)
aa[2].set_ylabel('$\delta[t]h[t-1]$')
aa[2].set_xlabel('$dE/dw=\sum\delta[t]h[t-1]=$%g'%(np.sum(grad)))
f.savefig('exp/fig%d.png'%(fignum))

######################################################################################3
# LSTM
fignum += 1
f = plt.figure(fignum,figsize=(6.4,8))
f.clf()
aa = f.subplots(nrows=4,ncols=1,sharex=True)
aa[0].stem(x,use_line_collection=True)
aa[0].set_title('Input x[t] of the LSTM')
aa[1].stem(y,use_line_collection=True)
aa[1].set_title('Target Output y[t] of the LSTM')
c = np.concatenate((np.cumsum(x[0:10]),np.cumsum(x[10:20]),np.cumsum(x[20:30])))
aa[2].stem(c,use_line_collection=True)
aa[2].set_title('Cell c[t] of the LSTM, f[t]=CReLU(1-h[t-1]), i[t]=CReLU(1)')
h = np.array([ c[t] if x[t]==0 else 0 for t in range(T) ])
aa[3].stem(h,use_line_collection=True)
aa[3].set_title('Output h[t] of the LSTM, o[t]=CReLU(1-x[t])')
f.savefig('exp/fig%d.png'%(fignum))

