import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os, requests
import scipy.stats as stats

os.makedirs('exp',exist_ok=True)

# Download data from the UCI repository
r=requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# Split at every newline; split each line at commas
dataset = [ x.split(',') for x in r.text.split('\n') ]
dataset[0:4]

################################################################################
# Get a dictionary from labels t indices, and back again
label2class = { 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
class2label = { 0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica' }
# Read out a list of the class labels, convert to integers
Y = [ label2class[x[4]] for x in dataset if len(x)==5 ]
# Plot the class labels of each token
fig, ax = plt.subplots()
ax.bar(range(0,len(Y)),Y)
plt.title('Class labels of all of the tokens')
plt.xlabel('Token number')
plt.ylabel('Class label')
fig.savefig('exp/classlabels.png')

################################################################################
# Create a numpy arrays for each data subset
X = np.array([ x[0:2] for x in dataset if len(x)==5 ], dtype='float64')
X.shape
fig, ax = plt.subplots()
# Plot a scatter plot of the three classes, so we can see how well they separate in the
# first two dimensions
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',
        X[100:150,0],X[100:150,1],'y^')
plt.title('Scatter plot of the three classes, in the first two dimensions')
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
fig.savefig('exp/scatter.png')

################################################################################
# Compute the mean of each class.  axis=0 means to compute the average row vector
mu = np.empty((3,2))
mu[0,:] = np.mean(X[0:50,:],axis=0)
mu[1,:] = np.mean(X[50:100,:],axis=0)
mu[2,:] = np.mean(X[100:150,:],axis=0)
fig, ax = plt.subplots()
ax.plot(mu[0,0],mu[0,1],'cx',mu[1,0],mu[1,1],'mo',mu[2,0],mu[2,1],'y^',markersize=18)
ax.plot(X[0:50,0],X[0:50,1],'c.',X[50:100,0],X[50:100,1],'m.',
        X[100:150,0],X[100:150,1],'y.')
ax.set_title('Mean vectors for each type of flower')
fig.savefig('exp/means.png')

################################################################################
# Compute the covariance of each class.
Sigma = np.empty((3,2,2))
Sigma[0,:,:] = np.cov(X[0:50,:],rowvar=False)
Sigma[1,:,:] = np.cov(X[50:100,:],rowvar=False)
Sigma[2,:,:] = np.cov(X[100:150,:],rowvar=False)

# Find the principal component directions and variances of each Gaussia
Lambda, V = np.linalg.eig(Sigma)

# Let's show the mean vectors plus/minus each variance-scaled principal component direction
fig, ax=plt.subplots()
ax.plot(X[0:50,0],X[0:50,1],'c.',X[50:100,0],X[50:100,1],'m.',
        X[100:150,0],X[100:150,1],'y.')
plt.title('Means +/- sqrt(eigenvalue)*(eigenvector)')
colors = 'cmy'
for y in range(0,3):
    for i in range(0,2):
        x0 = mu[y,0]+np.sqrt(Lambda[y,i])*np.array([-V[y,0,i],V[y,0,i]])
        x1 = mu[y,1]+np.sqrt(Lambda[y,i])*np.array([-V[y,1,i],V[y,1,i]])
        ax.plot(x0,x1,colors[y]+'-')
fig.savefig('exp/covariances.png')

################################################################################
# Create a coordinate system on which we can calculate the Gaussian pdf
x0coords, x1coords = np.mgrid[4:8:0.01,2:4.5:0.01]

# Create a coordinate system on which we can calculate the Gaussian pdf
coords = np.mgrid[4:8:0.01,2:4.5:0.01].transpose((1,2,0))

# Calculate the Gaussian pdfs at the same set of points
pdf = np.empty((400,250,3))
for y in range(0,3):
    pdf[:,:,y] = stats.multivariate_normal(mu[y,:],Sigma[y,:,:]).pdf(coords)
fig, ax = plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],np.amax(pdf,axis=2))
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('Contour plot of the three Gaussians, with scatter plot overlaid')

fig.savefig('exp/gaussian_contours.png')


################################################################################
# Find the (x0,x1) coordinates for which class 0 is the best choice
maxpdf = np.amax(pdf, axis=2)
yhat_is_0 = (pdf[:,:,0] == maxpdf)
yhat_is_1 = (pdf[:,:,1] == maxpdf)
yhat_is_2 = (pdf[:,:,2] == maxpdf)
yhat = yhat_is_1 + 2*yhat_is_2
# Now let's plot that
fig, ax = plt.subplots()
ax.contourf(x0coords,x1coords,yhat)
plt.title('Classifier output, yhat, plotted as a function of x')
fig.savefig('exp/gaussian_classifier.png')

################################################################################
# Plot just class #2
fig, ax = plt.subplots()
ax.contourf(x0coords,x1coords,pdf[:,:,2])
ax.plot(X[100:150,0],X[100:150,1],'y^')
ax.set_title('Gaussian  model of class #2 does not fit the data very well')
fig.savefig('exp/gaussian2_lackoffit.png')


################################################################################
# Train a GMM
# Initialize
mu_gmm = np.empty((3,2,2))
Sigma_gmm = np.empty((3,2,2,2))
c_gmm = np.empty((3,2))
fig, ax = plt.subplots()
plt.title('Initial mean vectors for the three Gaussian mixture models')
for y in range(0,3):
    # We need to find the maximum eigenvalue, because numpy doesn't sort them
    imax = np.argmax(Lambda[y,:])
    #imax=0
    for k in range(0,2):
        mu_gmm[y,k,:]=mu[y,:] + (-1)**k * 0.1*np.sqrt(Lambda[y,imax])*V[y,:,imax]
        Sigma_gmm[y,k,:,:]=Sigma[y,:,:]
        c_gmm[y,k] = 0.5
        ax.plot(mu_gmm[y,k,0],mu_gmm[y,k,1],colors[y]+'x')
fig.savefig('exp/gmm_initialmeans.png')


################################################################################
# Now let's calculate the GMM model of each of the three classes
gmm_gaussians = np.empty((400,250,3,2))
gmm_pdf = np.zeros((400,250,3))
for y in range(0,3):
    for k in range(0,2):
        gmm_gaussians[:,:,y,k] = stats.multivariate_normal(mu_gmm[y,k,:],Sigma_gmm[y,k,:,:]).pdf(coords)
        gmm_pdf[:,:,y] = gmm_pdf[:,:,y] + c_gmm[y,k]*gmm_gaussians[:,:,y,k]

fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],gmm_pdf[:,:,0])
ax.plot(X[0:50,0],X[0:50,1],'cx')
plt.title('Initial Gaussian Mixture Model of class 0, with scatter plot overlaid')
fig.savefig('exp/gmm_initialclass0.png')

################################################################################
fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],np.amax(gmm_pdf,axis=2))
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('Initial GMM of all three classes, with a scatter plot of the data overlaid')
fig.savefig('exp/gmm_initialcontours.png')

################################################################################
fig, ax = plt.subplots(3,1)
for y in range(0,3):
    # Perform K-means clustering for the y'th class
    # Start out with the dummy assumption that all data are in the 0'th cluster
    Xy = X[50*y:50*(y+1),0:2]
    k_index = np.zeros((50),dtype='int')
    # Find the resulting value of D, then set Dprev to 100 times that
    D = np.zeros((2))
    for n in range(0,50):
        D[-1] = D[-1] + np.sum(np.square(Xy[n,:]-mu_gmm[y,k_index[n],:]))
    D[0] = 1.5*D[-1]

    # Now iterate until convergence
    while D[-2]-D[-1] > 0.05*D[-2]:
        # Assign each datum to the nearest cluster
        for n in range(0,50):
            d0 = np.linalg.norm(Xy[n,0:2]-mu_gmm[y,0,:])
            d1 = np.linalg.norm(Xy[n,0:2]-mu_gmm[y,1,:])
            k_index[n]= np.argmin([d0,d1])
        # Now re-compute the centers (and the variances), based on the data assigned to each one
        for k in range(0,2):
            c_gmm[y,k] = np.count_nonzero(k_index==k)/50
            mu_gmm[y,k,:]=np.mean(Xy[k_index==k,:],axis=0)
            Sigma_gmm[y,k,:,:]=np.cov(Xy[k_index==k,:],rowvar=False)
        # Re-calculate D
        D = np.append(D,[0])
        for n in range(0,50):
            D[-1] = D[-1] + np.sum(np.square(Xy[n,:]-mu_gmm[y,k_index[n],:]))

    # After convergence, plot D
    ax[y].plot(D)
    ax[y].set_title('Convergence for class {}'.format(y))
fig.savefig('exp/kmeans_convergence.png')


################################################################################
fig, ax = plt.subplots()
for y in range(0,3):
    for k in range(0,2):
        ax.plot(mu_gmm[y,k,0],mu_gmm[y,k,1],colors[y]+'x')
ax.set_title('Means after convergence of the K-means algorithm')
fig.savefig('exp/kmeans_convergedmeans.png')

################################################################################
# Now let's calculate the GMM model of each of the three classes
gmm_gaussians = np.empty((400,250,3,2))
gmm_pdf = np.zeros((400,250,3))
for y in range(0,3):
    for k in range(0,2):
        gmm_gaussians[:,:,y,k] = stats.multivariate_normal(mu_gmm[y,k,:],Sigma_gmm[y,k,:,:]).pdf(coords)
        gmm_pdf[:,:,y] = gmm_pdf[:,:,y] + c_gmm[y,k]*gmm_gaussians[:,:,y,k]

fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],gmm_pdf[:,:,0])
ax.plot(X[0:50,0],X[0:50,1],'cx')
plt.title('K-means GMM class 0, with scatter plot overlaid')
fig.savefig('exp/kmeans_class0.png')

################################################################################
fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],np.amax(gmm_pdf,axis=2))
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('K-means GMM of all three classes, scatter plot overlaid')
fig.savefig('exp/kmeans_allclasses.png')

################################################################################
fig, ax=plt.subplots()
maxpdf = np.amax(gmm_pdf, axis=2)
yhat_using_em = (maxpdf == gmm_pdf[:,:,1]) + 2*(maxpdf == gmm_pdf[:,:,2])
ax.contourf(coords[:,:,0],coords[:,:,1],yhat_using_em)
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('GMM classification function from K-means, scatter plot overlaid')
fig.savefig('exp/kmeans_classifier.png')

################################################################################
fig, ax = plt.subplots(3,1)
for y in range(0,3):
    # Perform Expectation Maximization for the y'th class
    # Start out with the parameters from k-means
    c_em = c_gmm
    mu_em = mu_gmm
    Sigma_em = Sigma_gmm
    Xy = X[50*y:50*(y+1),0:2]
    data_loglikelihood = []
    
    # Now iterate ten times (we won't check convergence, but we'll check how the data log likelihood evolves)
    for iteration_count in range(0,10):
        # Create an array of the random variables
        rvs = [ stats.multivariate_normal(mu_em[y,0,:],Sigma_em[y,0,:,:]),
                  stats.multivariate_normal(mu_em[y,1,:], Sigma_em[y,1,:,:]) ]

        # Accumulate sufficient statistics for the numerator of each parameter's M-step
        c_numerator = np.zeros(2)
        mu_numerator = np.zeros((2,2))
        Sigma_numerator = np.zeros((2,2,2))
                                
        # E-step: calculate the gamma probability, and the sufficient statistics
        gamma_denom = np.zeros(50)
        for n in range(0,50):
            gamma_denom[n] = np.inner(c_em[y,:], [rvs[0].pdf(Xy[n,:]), rvs[1].pdf(Xy[n,:]) ] )
            for k in range(0,2):
                gamma = c_em[y,k]*rvs[k].pdf(Xy[n,:])/gamma_denom[n]
                c_numerator[k] = c_numerator[k] + gamma
                mu_numerator[k,:] = mu_numerator[k,:] + gamma*Xy[n,:]
                Sigma_numerator[k,:,:] = Sigma_numerator[k,:,:] + gamma*np.outer(Xy[n,:],Xy[n,:])

        # M-step: re-compute the parameters
        for k in range(0,2):
            c_em[y,k] = c_numerator[k] / 50
            mu_em[y,k,:] = mu_numerator[k,:] / c_numerator[k]
            Sigma_em[y,k,:,:] = Sigma_numerator[k,:,:] / c_numerator[k] - np.outer(mu_em[y,k,:],mu_em[y,k,:])
            # Make sure sigma is positive definite
            d = np.linalg.det(Sigma_em[y,k,:,:])
            if d < 0.01:
                sqrtd = np.sqrt(0.01-d)
                Sigma_em[y,k,:,:] = Sigma_em[y,k,:,:] + sqrtd*np.eye(2)
                
        # Re-calculate log likelihood
        data_loglikelihood.append( np.sum(np.log(gamma_denom)) )

    # After convergence, plot data_loglikelihood
    ax[y].plot(np.arange(1,len(data_loglikelihood)),data_loglikelihood[1:])
    ax[y].set_title('Data Log Likelihood vs. iteration number for class {}'.format(y))
fig.savefig('exp/em_convergence.png')

################################################################################
# Now let's calculate the GMM model of each of the three classes
em_gaussians = np.empty((400,250,3,2))
em_pdf = np.zeros((400,250,3))
for y in range(0,3):
    for k in range(0,2):
        em_gaussians[:,:,y,k] = stats.multivariate_normal(mu_em[y,k,:],Sigma_em[y,k,:,:]).pdf(coords)
        em_pdf[:,:,y] = em_pdf[:,:,y] + c_em[y,k]*em_gaussians[:,:,y,k]

fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],em_pdf[:,:,2])
ax.plot(X[100:150,0],X[100:150,1],'y^')
plt.title('GMM contour plot of class 2, trained using EM')
fig.savefig('exp/em_class2.png')

################################################################################
fig, ax = plt.subplots()
markertype='xo^'
for y in range(0,3):
    for k in range(0,2):
        ax.plot(mu_em[y,k,0],mu_em[y,k,1],colors[y]+markertype[y],markersize=18)
ax.plot(X[0:50,0],X[0:50,1],'c.',X[50:100,0],X[50:100,1],'m.',X[100:150,0],X[100:150,1],'y.')
ax.set_title('Means after convergence of the EM algorithm')
fig.savefig('exp/em_convergedmeans.png')

################################################################################
fig, ax=plt.subplots()
ax.contourf(coords[:,:,0],coords[:,:,1],np.amax(em_pdf,axis=2))
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('GMM contour plots of all three classes, trained using EM')
fig.savefig('exp/em_allcontours.png')

################################################################################
fig, ax=plt.subplots()
maxpdf = np.amax(em_pdf, axis=2)
yhat_using_em = (maxpdf == em_pdf[:,:,1]) + 2*(maxpdf == em_pdf[:,:,2])
ax.contourf(coords[:,:,0],coords[:,:,1],yhat_using_em)
ax.plot(X[0:50,0],X[0:50,1],'cx',X[50:100,0],X[50:100,1],'mo',X[100:150,0],X[100:150,1],'y^')
plt.title('GMM classification function from EM, scatter plot overlaid')
fig.savefig('exp/em_classifier.png')


