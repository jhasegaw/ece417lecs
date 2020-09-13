import numpy as np
import math
import matplotlib.figure
import matplotlib.gridspec
from matplotlib.patches import FancyArrowPatch

def vec2line(w, b, xmin, xmax):
    '''
    return (x0, x1) s.t.  x0w[0]+x1w[1]+b = 0
    and s.t. each (x0,x1) pair is on one of the four  boundaries x0min, x0max, x1min, x1max.
    '''
    x0 = np.array([xmin[0],xmax[0]])
    x1 = np.minimum(xmax[1],np.maximum(xmin[1],(-b-x0*w[0])/w[1]))  # Find x1, then trunc at x1min, x1max
    x0 = (-b-x1*w[1])/w[0] # Recompute x0, to see if it's changed
    if np.any(x0 < xmin[0])  or np.any(x0 > xmax[0]):    # If any resulting x are out of bounds, return 0
        return(np.zeros(2),np.zeros(2))
    return(x0,x1)

def sigmoid(x):
    return(1/(1+np.exp(-x)))

###################################################################
# Standard figures
def plot_xdata(ax, X, xmin, xmax, W):
    # data
    ax.plot(X[0],X[1],'rD')
    # axes
    ax.plot([xmin[0],xmax[0]],[0,0],'k-',[0,1e-6],[xmin[1],xmax[1]],'k-') # axes
    # unit circle
    ax.plot(np.cos(np.linspace(-np.pi/2,np.pi/2,100)),np.sin(np.linspace(-np.pi/2,np.pi/2,100)),'k--')
    # weights
    for k in range(W.shape[1]):
        x0,x1 = vec2line(W[:,k],0,xmin,xmax)
        ax.plot(x0,x1,'g--')
    ax.set_title('Input (x) Plane')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
def plot_ydata(ax, Y, ymin, ymax):
    ax.plot(Y[0],Y[1],'rD') # data
    ax.plot([ymin[0],ymax[0]],[0,0],'k-',[0,1e-6],[ymin[1],ymax[1]],'k-') # axes
    ax.plot(np.linspace(-np.pi/2,np.pi/2,100),np.power(np.linspace(-np.pi/2,np.pi/2,100),2),'k--')
    ax.set_title('Output (y) Plane')
    ax.set_xlabel('$y_0$')
    ax.set_ylabel('$y_1$')
def plot_e_or_h(ax, E, title):
    ax.bar(np.arange(len(E)), E)
    ax.set_title(title)

###################################################################
# example: y1=atan(x2/x1), y2=atan(x2/x1)^2, angles from -pi/2 to pi/2
thresholds = -np.pi/2 + (np.pi/16) + np.arange(0,8)*(np.pi/8)
theta = -np.pi/2 + np.arange(0,9)*(np.pi/8)
W1i = np.concatenate((-np.expand_dims(np.sin(thresholds),0), np.expand_dims(np.cos(thresholds),0)),0)
Xi = np.concatenate((np.expand_dims(np.cos(theta),0), np.expand_dims(np.sin(theta),0)),0)
Yi = np.concatenate((np.expand_dims(theta,0), np.expand_dims(np.power(theta,2),0)),0)
Bi = Yi[:,0]
W2i = np.diff(Yi, axis=1)

#xmin = np.amin(Xi,axis=1)-0.1
xmin = np.array([-1.1,-1.1])
#xmax = np.amax(Xi,axis=1)+0.1
xmax = np.array([1.1,1.1])
ymin = np.amin(Yi,axis=1)-0.1
ymax = np.amax(Yi,axis=1)+0.1
    
###########################################################################
# Iamge: Show the x plane, and the y plane
# show dashed lines in the x plane, and corresponding x marks in the y plane?
fig = matplotlib.figure.Figure(figsize=(10, 4))
axs = fig.subplots(1,2)
plot_xdata(axs[0], Xi, xmin, xmax, W1i)
plot_ydata(axs[1], Yi, ymin, ymax)
fig.savefig('exp/nn_target_figure.png')

###########################################################################
# Video: show
# 1. x passes one of the dashed lines (radius random between 0.75 and 1.25)
#     this subplot also shows w^(1) of the vector that x just passed, as a vector
# 2a. a column chart of x^T w^(1)_k for each k, showing some positive, some still negative
# 2b. a column chart of u(x^T w^(1)_k), showing some are 1, others 0
# 3. the y plane: show y moving from one 'x' to the next, as each new w^(2) is added
#   title? shows y = b + ... + w^(2)_k , k varying as u() varies
def arrowseq(ax, W):
    S = np.cumsum(W,axis=1)
    patch = FancyArrowPatch((0,0),S[:,0],mutation_scale=10)
    patch.set_color('blue')
    ax.add_patch(patch)
    for k in range(1,W.shape[1]):
        patch = FancyArrowPatch(S[:,k-1],S[:,k],mutation_scale=10)
        patch.set_color('blue')
        ax.add_patch(patch)
    return(S)

def plot_all_arrows(ax, W):
    for k in range(1,W.shape[1]):
        ax.add_patch(FancyArrowPatch((0,0),W[:,k],mutation_scale=10))
        
def plot_forward(axs, X, Y, W1, B, W2, k, E=None, H=None):
    for a in axs:
        a.clear()
    plot_xdata(axs[0], X, xmin, xmax, W1)
    if E is None:
        E = np.inner(W1.T,X)
    plot_e_or_h(axs[1], E, 'Excitation $e=x^T W^{(1)}$')
    axs[1].set_ylim([-1.1,1.1])    
    if H is None:
        H = sigmoid(E)
    plot_e_or_h(axs[2], H,'')
    axs[2].plot([-1,len(H)+1],[0.5,0.5],'k--')
    axs[2].set_ylim([-0.1,1.1])
    axs[2].set_xlabel('Activation $h=\sigma(e)$')
    plot_ydata(axs[3], Y, ymin, ymax)
    a = np.zeros((2,k+1))
    a[:,0] = B
    a[:,1:(k+1)] = W2[:,0:k] * np.tile(np.heaviside(E[0:k],1),[2,1])
    S = arrowseq(axs[3], a)
    axs[3].set_title('$ŷ=b+...+w^{(2)}_%d$'%(k))
    return(E,H, S)
    
fig = matplotlib.figure.Figure(figsize=(14, 4))
gs = matplotlib.gridspec.GridSpec(2,3,figure=fig)
axs = []
axs.append(fig.add_subplot(gs[:,0]))
axs.append(fig.add_subplot(gs[0,1]))
axs.append(fig.add_subplot(gs[1,1]))
axs.append(fig.add_subplot(gs[:,2]))

for k in range(Xi.shape[1]):
    E,H,S = plot_forward(axs, Xi[:,k], Yi[:,k], W1i, Bi, W2i, k)
    if k > 0:
        patch=FancyArrowPatch((0,0), W1i[:,k-1], mutation_scale=20)
        patch.set_color('blue')
        axs[0].add_patch(patch)
        x0,x1 = vec2line(W1i[:,k-1],0,xmin,xmax)
        axs[0].plot(x0,x1,'b-')
    else:
        axs[3].set_title('$ŷ=b$')
    fig.savefig('exp/nnapprox%d.png'%(k))

###########################################################################
# Video of training
rng = np.random.default_rng()
W1r = 0.5*rng.standard_normal((2,5)) + np.tile(np.array([[0.5],[0]]),[1,5])
W2r = 0.5*rng.standard_normal((2,5)) + np.tile(np.array([[0.32],[0]]),[1,5])
B2r = 0.5*rng.standard_normal(2) + np.array([-0.32,1.5])
# An x and y are presented.
# dashed lines show locations of w^(1) in x plane, randomly initialized (only about 5, maybe).
# column charts show x^T w^(1)_k and sigma(x^T w^(1)_k).
datum = rng.integers(0,W1r.shape[1])
E,H, S = plot_forward(axs, Xi[:,datum], Yi[:,datum], W1r, B2r, W2r, 5)
Yhat = S[:,-1]
plot_all_arrows(axs[0], W1r)
axs[0].set_title('Training Datum (x) and Random Initial $W^{(1)}$')
patch = FancyArrowPatch(Yi[:,datum],Yhat,mutation_scale=10)
patch.set_color('red')
axs[3].add_patch(patch)
axs[3].set_title('Error: $\epsilon=ŷ-y$')
fig.savefig('exp/nntrain_init.png')

err = Yhat - Yi[:,datum]
delta = np.inner(err, W2r.T)
dW1 = - np.outer(Xi[:,datum], delta * H * (1-H))
dW2 = - np.outer(err, H)
for k in range(5):
    for t in range(100):
        E,H,S = plot_forward(axs, Xi[:,datum], Yi[:,datum], W1r, B2r, W2r, 5, E=E, H=H)
        patch = FancyArrowPatch(Yi[:,datum],Yhat,mutation_scale=10)
        patch.set_color('red')
        axs[3].add_patch(patch)
        #   delta_k sigma'(x^T w^(1)_k) x, subtracted from the end of w^(1)_k as update.
        axs[0].add_patch(FancyArrowPatch(W1r[:,k],W1r[:,k]+dW1[:,k],mutation_scale=10))
        #   yhat-y is multiplied by h_k, and subtracted from the end of w^(2)_k to show its update.
        axs[3].add_patch(FancyArrowPatch(S[:,k],S[:,k]+dW2[:,k],mutation_scale=10))
        # Here are the parts that move, over time
        W1t = W1r[:,k]+0.01*t*dW1[:,k]
        axs[0].add_patch(FancyArrowPatch((0,0),W1t,mutation_scale=10))
        x0,x1 = vec2line(W1t,0,xmin,xmax)
        axs[0].plot(x0,x1,'b-')
        W2t = S[:,k]+0.01*t*dW2[:,k]
        if k==0:
            axs[3].add_patch(FancyArrowPatch((0,0),W2t,mutation_scale=10))
        else:
            axs[3].add_patch(FancyArrowPatch(S[:,k-1],W2t,mutation_scale=10))
        axs[0].set_title("$\delta w^{(1)}_%d=-x\sigma'(e_%d)\epsilon^Tw^{(2)}_%d$"%(k,k,k))
        axs[3].set_title("$\delta w^{(2)}_%d=-h_%d\epsilon$"%(k,k))
        fig.savefig('exp/nntrain%d.png'%(100*k+t))
    # After the video is made, change W1r and W2r
    W1r[:,k] += dW1[:,k]
    W2r[:,k] += dW2[:,k]
    
