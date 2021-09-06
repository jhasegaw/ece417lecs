import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os

os.makedirs('exp',exist_ok=True)

######################################################################################
# Delta function
n1 = np.repeat(np.arange(-10,11).reshape((21,1)), 21, axis=1)
n2 = np.repeat(np.arange(-10,11).reshape((1,21)), 21, axis=0)
delta = np.zeros((21,21))
delta[10,10] = 1
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(n1, n2, delta)
ax.set_xlabel('$n_1$')
ax.set_ylabel('$n_2$')
ax.set_title('$\delta[n_1,n_2]$')
fig.savefig('exp/delta.png')

######################################################################################
# Fourier
fourier = plt.imread('exp/fourier.jpg')
N1,N2 = fourier.shape
n1 = np.repeat(np.arange(N1).reshape((N1,1)), N2, axis=1)
n2 = np.repeat(np.arange(N2).reshape((1,N2)), N1, axis=0)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(n1, n2, fourier)
fig.savefig('exp/fourier_2dmesh.png')

######################################################################################
# Fourier Transformed
F = np.fft.fftshift(np.fft.fft2(fourier))
omega1 = np.repeat((np.arange(-int(N1/2),int((N1+1)/2))*2*np.pi/N1).reshape((N1,1)),N2,axis=1)
omega2 = np.repeat((np.arange(-int(N2/2),int((N2+1)/2))*2*np.pi/N2).reshape((1,N2)),N1,axis=0)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(omega1, omega2, np.abs(F))
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
ax.set_title('$|F(\omega_1,\omega_2)|$')
fig.savefig('exp/fourier_transformed.png')
fig, ax = plt.subplots()
ax.imshow(np.abs(F),extent=[-np.pi,np.pi,-np.pi,np.pi])
ax.set_title('Fourier Transformed')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
fig.savefig('exp/fourier_transformed_image.png')

######################################################################################
# Gauss
m1 = np.repeat(np.arange(-5,6).reshape((11,1)), 11, axis=1)
m2 = np.repeat(np.arange(-5,6).reshape((1,11)), 11, axis=0)
sigma = 1.5
var = sigma*sigma 
g2d = np.exp(-0.5*(m1*m1+m2*m2)/var)/(2*np.pi*var)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(m1, m2, g2d)
ax.set_xlabel('$m_1$')
ax.set_ylabel('$m_2$')
ax.set_title('Gaussian Blur kernel $g[m_1,m_2]$, with $\sigma=1.5$')
fig.savefig('exp/gauss_2d.png')

# 1d
g1d = np.exp(-0.5*m1[:,5]*m1[:,5]/var)/np.sqrt(2*np.pi*var)
fig, ax = plt.subplots()
plt.stem(m1[:,5], g1d)
ax.set_xlabel('$m_1$')
ax.set_title('Gaussian Blur kernel $g[m_1]$ in 1d, with $\sigma=1.5$')
fig.savefig('exp/gauss_1d.png')

# transform
G2D = np.fft.fftshift(np.fft.fft2(g2d, s=(N1,N2)))
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(omega1, omega2, np.abs(G2D))
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
ax.set_title('Gaussian Blur $|G(\omega_1,\omega_2)|$')
fig.savefig('exp/gauss_transformed.png')

# filter Fourier
tmp = np.transpose(np.array([ np.convolve(g1d, x) for x in fourier ]))
smoothed = np.transpose(np.array([ np.convolve(g1d, x) for x in tmp ]))
fig, ax = plt.subplots()
plt.imshow(smoothed, cmap='gray',extent=[-5,5,-5,5], aspect='equal')
ax.set_xlabel('$n_1$')
ax.set_ylabel('$n_2$')
ax.set_title('Smoothed image of Fourier')
fig.savefig('exp/fourier_smoothed.png')

######################################################################################
# Non-separable filter
nonsep = np.zeros((11,11))
nonsep[3:8,5]=1
nonsep[5,3:8]=1
nonsep[4,4:7]=1
nonsep[6,4:7]=1
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(m1, m2, nonsep)
ax.set_xlabel('$m_1$')
ax.set_ylabel('$m_2$')
ax.set_title('Diamond Window $w[m_1,m_2]$')
fig.savefig('exp/diamond.png')
fig, ax = plt.subplots()
plt.imshow(nonsep, cmap='gray')
plt.title('Diamond Window $w[m_1,m_2]$')
fig.savefig('exp/diamond_image.png')
