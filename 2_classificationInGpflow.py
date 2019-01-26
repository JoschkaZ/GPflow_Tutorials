
# domonstration of sparse variational GP classification on simple dataset
# TODO sparse?
# TODO variational ?

from matplotlib import pyplot as plt
import sys
import csv
import numpy as np
import gpflow
import time

# load data
Xtrain = np.loadtxt('data/banana/banana_X_train', delimiter=',')
Ytrain = np.loadtxt('data/banana/banana_Y_train', delimiter=',').reshape(-1,1)
# Xtrain has 2 features and N samples
# Ytrain has a label (0 or 1) and N samples


def gridParams():
    mins = [-3.25,-2.85 ]
    maxs = [ 3.65, 3.4 ]
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(),yy.flatten())).T
    return mins, maxs, xx, yy, Xplot

def plot(m, ax):
    col1 = '#0172B2'
    col2 = '#CC6600'
    mins, maxs, xx, yy, Xplot = gridParams()
    p = m.predict_y(Xplot)[0]
    ax.plot(Xtrain[:,0][Ytrain[:,0]==1], Xtrain[:,1][Ytrain[:,0]==1], 'o', color=col1, mew=0, alpha=0.5)
    ax.plot(Xtrain[:,0][Ytrain[:,0]==0], Xtrain[:,1][Ytrain[:,0]==0], 'o', color=col2, mew=0, alpha=0.5)
    if hasattr(m, 'feature') and hasattr(m.feature, 'Z'):
        Z = m.feature.Z.read_value()
        ax.plot(Z[:,0], Z[:,1], 'ko', mew=0, ms=4)
        ax.set_title('m={}'.format(Z.shape[0]))
    else:
        ax.set_title('full')
    ax.contour(xx, yy, p.reshape(*xx.shape), [0.5], colors='k', linewidths=1.8, zorder=100)



# setup experiment and plotting
Ms = [2,4, 8, 16, 32, 64, 128, 256] #number of inducing points used

# Run sparse classification with increasing number of inducing points
models = []
times = []

for index, num_inducing in enumerate(Ms):
    start = time.time()
    # use kmeans for selecting Z
    from scipy.cluster.vq import kmeans
    Z = kmeans(Xtrain, num_inducing)[0] #this returns the n clustered centers of the data ?
    print('\nZ shape: ', Z.shape)

    # TODO need to look up what SVGP actually does...
    m = gpflow.models.SVGP(
    Xtrain, Ytrain, kern=gpflow.kernels.RBF(2),
    likelihood=gpflow.likelihoods.Bernoulli(), Z=Z)
    print('\n Model:\n', m)

    # initially fix hyperparameters
    m.feature.set_trainable(False)
    # this sets Z to false
    # kernel lengthscale, kernel variance, q_mu and q_sqrt are still trained
    gpflow.train.ScipyOptimizer().minimize(m, maxiter=20)

    # unfix
    m.feature.set_trainable(True)
    gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(m)
    print('\n After Fitting\n', m)
    models.append(m)
    times.append(time.time()-start)


# Run variational approximation without sparsity..
# ..be aware that this is much slower for big datasets,
# but relatively quick here.
m = gpflow.models.VGP(Xtrain, Ytrain,
                      kern=gpflow.kernels.RBF(2),
                      likelihood=gpflow.likelihoods.Bernoulli())
gpflow.train.ScipyOptimizer().minimize(m, maxiter=2000)
models.append(m)


# make plots.
fig, axes = plt.subplots(1, len(models), figsize=(12.5, 2.5), sharex=True, sharey=True)
for i, m in enumerate(models):
    plot(m, axes[i])
    axes[i].set_yticks([])
    axes[i].set_xticks([])
plt.show()

plt.plot(times)
plt.show()
