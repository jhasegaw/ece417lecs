import numpy as np
import matplotlib.figure,subprocess,os

os.makedirs('exp',exist_ok=True)

########################################################################
# Main idea: h0=2+cos(x), h1=2+sin(x), yhat=sqrt(h0^2+h1^2)
# Show dyhat/dh0, then dyhat/dh1, then dyhat/dtheta
x = np.linspace(0,10*np.pi,100,endpoint=False)
h0 = np.cos(x)
h1 = (h0*h0*h0 +  np.sin(x))/np.sqrt(2)
yhat = np.sqrt(h0*h0+h1*h1)

def plot2(ax,h0,h1):
    ax.clear()
    ax.plot([-1.1,1.1],[0,0],'k-',[0,1e-6],[-1.1,1.1],'k-')
    ax.plot(h0+[-(1e-6),0],[0,h1],'b-')
    ax.plot([0,h0],[h1-(1e-6),h1],'b-')
    ax.plot([0,h0],[0,h1],'r-')
    ax.set_xlabel('$h_0(x)$')
    ax.set_ylabel('$h_1(x)$')
    ax.set_title('$y(x)=sqrt(h_0^2(x)+h_1^2(x))$')

def xplot(ax,x):
    ax.clear()
    ax.plot([-0.1,N+0.1],[0,0],'k-',[0,1e-6],[-np.pi-0.1,np.pi+0.1],'k-')
    ax.plot(np.arange(len(x)),x,'b-')
    ax.set_title('$x[n]$')

def hplot(ax,x,h,title):
    ax.clear()
    ax.plot([-0.1,10*np.pi+0.1],[0,0],'k-',[0,1e-6],[-1.1,1.1],'k-')
    ax.plot(x,h,'b-')
    ax.set_title(title)
    
def yplot(ax,x,y):
    ax.clear()
    ax.plot([-0.1,10*np.pi+0.1],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
    ax.plot(x,y,'r-')
    ax.set_title('$y(x)=sqrt(h_0^2(x)+h_1^2(x))$')
    ax.set_xlabel('$x$')

#############################################
# h0step: Show h0 and yhat as functions of theta, with no variation of h1

fig = matplotlib.figure.Figure((8,4))
gs = fig.add_gridspec(3,6)
axs = [ fig.add_subplot(gs[:,0:3]), fig.add_subplot(gs[0,3:]),
        fig.add_subplot(gs[1,3:]), fig.add_subplot(gs[2,3:]) ]
for n in range(len(x)):
    plot2(axs[0],h0[n],np.ones(1))
    hplot(axs[1],x[:n],h0[:n],'$h_0(x)=cos(x)$')
    hplot(axs[2],x[:n],np.ones(n),'$h_1(x)=1$')
    yplot(axs[3],x[:n],np.sqrt(h0[:n]*h0[:n]+1))
    fig.tight_layout()
    fig.savefig('exp/h0step%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/h0step?.png exp/h0step??.png exp/h0step.gif'.split())

for n in range(len(x)):
    plot2(axs[0],h0[n],h1[n])
    hplot(axs[1],x[:n],h0[:n],'$h_0(x)=cos(x)$')
    hplot(axs[2],x[:n],h1[:n],'$h_1(x)=(h_0^3(x)+sin(x))/√2$')
    yplot(axs[3],x[:n],yhat[:n])
    fig.tight_layout()
    fig.savefig('exp/h1step%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/h1step?.png exp/h1step??.png exp/h1step.gif'.split())

