import numpy as np
import math,subprocess
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm


os.makedirs('exp',exist_ok=True)

#####################################################################
# Schematic of gradient descent

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = sigmoid(2-X)+sigmoid(2-Y)+sigmoid(1+X)+sigmoid(1+Y)
dZdX = sigmoid_deriv(2-X)+sigmoid_deriv(1+X)
dZdY = sigmoid_deriv(2-Y)+sigmoid_deriv(1+Y)
print(dZdX[20,20])
print(dZdY[20,20])

# Plot the arrow
a = Arrow3D([0,-2], [0,-2],
            [Z[20,20]+0.25, Z[10,10]+0.75], mutation_scale=20,
            lw=5, arrowstyle="-|>", color="g")
ax.add_artist(a)
x = np.linspace(0,-2,11)
z = np.array([Z[m,m]+0.1 for m in np.arange(20,9,-1)])
#ax.plot(x,x,z,'ro', alpha=0.5)
surf = ax.plot_surface(X, Y, Z, cmap=cm.gray, linewidth=0, antialiased=False)
ax.set_title('One step of gradient descent on a complicated error surface')
ax.set_zlabel('Error')
ax.set_xlabel('Parameter 1')
ax.set_xlabel('Parameter 2')
fig.tight_layout()
fig.savefig('exp/gradient_descent.png')

exit()

#####################################################################
# Video of forward propagation

# Input channels are RGB, output channels are CMYK
rcx = np.array([0,0,5,5,5,5,0,0,0,5,5,5,5,0,0,0])
rcy = np.array([0,0,0,0,5,5,5,5,0,0,0,5,5,5,5,0])
rcz = np.array([0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1])
def plot_channels(ax):
    ax.plot(rcx,rcy,rcz,'r-')
    ax.plot(rcx,rcy,rcz+1,'g-')
    ax.plot(rcx,rcy,rcz+2,'b-')
    ax.plot(rcx,rcy,rcz+10,'c-')
    ax.plot(rcx,rcy,rcz+11,'m-')
    ax.plot(rcx,rcy,rcz+12,'y-')
    ax.plot(rcx,rcy,rcz+13,'k-')

# Weights Forward are 3x3x3, Weights backward are 3x3x4
wfx = np.array([0,0,3,3,3,3,0,0,0,3,3,3,3,0,0,0])-1.5
wfy = np.array([0,0,0,0,3,3,3,3,0,0,0,3,3,3,3,0])-1.5
wfz = np.array([0,3,3,0,0,3,3,0,0,0,3,3,0,0,3,3])
# Triangle used to connect output point to input filter
tx = np.array([0,-1.5,0,1.5,0,1.5,0,-1.5,-1.5,1.5,1.5,-1.5])
ty = np.array([0,-1.5,0,-1.5,0,1.5,0,1.5,-1.5,-1.5,1.5,1.5])
tz = np.array([0,1.5,0,1.5,0,1.5,0,1.5,1.5,1.5,1.5,1.5])/1.5
def plot_forward(ax,x,y,z,color):
    ax.plot(wfx+x,wfy+y,wfz,color+'-')
    ax.plot(tx+x,ty+y,z*(1-tz),color+'-')

def plot_backward(ax,x,y,z,color):
    ax.plot(wfx+x,wfy+y,(4/3)*wfz+10,color+'-')
    ax.plot(tx+x,ty+y,z+(10-z)*tz,color+'-')

# Points
pxf = np.tile(np.array([0,1,2,3,4]*5).reshape((1,25)),(4,1))+0.5
pyf = np.tile(np.array([0]*5+[1]*5+[2]*5+[3]*5+[4]*5).reshape((1,25)),(4,1))+0.5
pzf = np.concatenate((np.zeros((1,25)),np.ones((1,25)),2*np.ones((1,25)),3*np.ones((1,25))))+10.5

# Forward video
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
cf = 'cmyk'
for k in range(4):
    for n in range(25):
        ax.clear()
        plot_channels(ax)
        plot_forward(ax,pxf[k,n],pyf[k,n],pzf[k,n],cf[k])
        for c in range(k): 
            ax.scatter(pxf[c,:],pyf[c,:],pzf[c,:],c=cf[c],marker='^')
        ax.scatter(pxf[k,0:(n+1)],pyf[k,0:(n+1)],pzf[k,0:(n+1)],c=cf[k],marker='^')
        ax.set_xlim((-1.5,6.5))
        ax.set_ylim((-1.5,6.5))
        ax.set_zlim((-0.5,14.5))
        ax.set_zticks([1.5,12])
        ax.set_zticklabels(['   Layer 0','  Layer 1'])
        fig.savefig('exp/forward%d.png'%(25*k+n))
subprocess.call('convert -delay 25 -dispose previous exp/forward?.png exp/forward??.png  lec20_forward.gif'.split())

#####################################################################
# Video of backpropagation

# Backward video
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
pxb = np.tile(np.array([0,1,2,3,4]*5).reshape((1,25)),(3,1))+0.5
pyb = np.tile(np.array([0]*5+[1]*5+[2]*5+[3]*5+[4]*5).reshape((1,25)),(3,1))+0.5
pzb = np.concatenate((np.zeros((1,25)),np.ones((1,25)),2*np.ones((1,25)))) + 0.5
cb = 'rgb'
for k in range(3):
    for n in range(25):
        ax.clear()
        plot_channels(ax)
        plot_backward(ax,pxb[k,n],pyb[k,n],pzb[k,n],cb[k])
        for c in range(k): 
            ax.scatter(pxb[c,:],pyb[c,:],pzb[c,:],c=cb[c],marker='v')
        ax.scatter(pxb[k,0:(n+1)],pyb[k,0:(n+1)],pzb[k,0:(n+1)],c=cb[k],marker='v')
        ax.set_xlim((-1.5,6.5))
        ax.set_ylim((-1.5,6.5))
        ax.set_zlim((-0.5,14.5))
        ax.set_zticks([1.5,12])
        ax.set_zticklabels(['   Backprop to 0','  Backprop to 1'])
        fig.savefig('exp/backward%d.png'%(25*k+n))
subprocess.call('convert -delay 25 -dispose previous exp/backward?.png exp/backward??.png  lec20_backward.gif'.split())
