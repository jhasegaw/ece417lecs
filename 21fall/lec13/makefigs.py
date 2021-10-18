import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('exp',exist_ok=True)

fig, ax = plt.subplots(figsize=(12,3))

xfunc = np.linspace(0.5,5,100)
yfunc = np.log(xfunc)
ax.plot(xfunc,yfunc,'b-')   # the log curve
ax.plot([-1,5],[0,0],'k--',[0,1e-6],[-np.log(2),np.log(5)+0.5],'k--')  # X and Y axes
x1 = 0.75
x2 = 3.75
y1 = np.log(x1)
y2 = np.log(x2)
ax.plot([x1,x2],[y1,y2],'g-')                      # secant line at x1 and x2
x0 = 0.4*x1+0.6*x2
y0 = 0.4*np.log(x1)+0.6*np.log(x2)
x3 = x1+x2
y3 = np.log(x3)
ax.plot(x0,y0,'ro',x3,y3,'ro')
xticklabels = ['$\sum_k p(k)x(k)$','$x(1)$', '$x(2)$','$\sum_k x(k)$']
yticklabels = ['$\sum_k p(k)\ln x(k)$','$\ln x(1)$', '$\ln x(2)$','$\ln\sum_k x(k)$']
for (x,y,xs,ys) in zip([x0,x1,x2,x3],[y0,y1,y2,y3],xticklabels,yticklabels):
    if x==x0 or x==x3:
        ax.plot([0,x],y*np.ones(2),'r--')  # horizontal ID line
        ax.text(-0.1,y,ys,horizontalalignment='right',verticalalignment='center',fontsize=24)
    ax.plot([x,x+1e-6],[0,y],'r--')  # vertical ID line
    if x==x1:
        ax.text(x,0.01,xs,horizontalalignment='center',verticalalignment='bottom',fontsize=24)
    else:
        ax.text(x,-0.01,xs,horizontalalignment='center',verticalalignment='top',fontsize=24)

ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('exp/eminequality.png')


