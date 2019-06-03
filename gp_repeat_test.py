import sys
import numpy as np
import numpy.random as rnd
import time
import gpflow
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import copy
import scipy.stats as st
from scipy.stats import t
from scipy.stats import norm


train_features = np.array([
[1.],
[2.],
[3.]
])

Z = train_features

train_labels = np.array([
[1.],
[2.],
[1.]
])



kern = (
gpflow.kernels.RBF(
    input_dim = 1, # number of dimensions
    variance = 1., # initial value for varance parameter (magnitude)
    lengthscales = 1., # if 1:no ARD, if np.ones(input_dim): yes ARD)
    active_dims = None,  # list of length input_dim: which columns of X are used
    ARD = True)) # if true: one lengthscale per dimension, if false: single lengthscale, otherwise infered

likelihood = gpflow.likelihoods.Gaussian(variance=1)

m = gpflow.models.SVGP(
    X = train_features, # data matrix (N x D) # dataholder object, if minibatch_size not not: Minibatch object (inherits from DataHolder)
    Y = train_labels, # data matrix (N x P)
    kern = kern, # a kernel object
    likelihood = likelihood, # a likelihood object
    feat = None, # no idea
    mean_function = None, # a gpflow object (maps X to Y (for what?))
    num_latent = None, # number of latent process to use (default: dim of Y)
    q_diag = False, # if true: covariance is approximated with diagonal matrix
    whiten = True, # if true: whitened representation of inducing points is used
    minibatch_size = 3, # if not none: uses minibatches with given size
    Z = Z, # matrix of pseudo inputs (M x D)
    num_data = None, # total number of observations (needed if external mini batches are used)
    q_mu = None, # mean and cholesky of covariance of variational gaussian posterior is constructed. if not none: it is checked if they have consistent shapes. if not specified they are initialized based on num_inducing and q_diag
    q_sqrt = None # see above
)

m.likelihood.variance.trainable = False
#m.likelihood.scale.trainable = False
m.feature.trainable = False
m.kern.lengthscales.trainable = False

class Logger(gpflow.actions.Action): #inherits from Action class
    # NOTE ! THE MODEL IN THE LOGGER IS ONLY UP TO DATA AFTER ANCHORING
    # action is a metaclass for wrapping functions in a container
    # action has a watcher used to measure time spent on execution
    def __init__(self, model):
        self.model = model
        self.logf = [] # logs the likelihood every 10 iterations
        self.best_likelihood = None
        self.early_stopping_count = None
        self.param_hist = []

    def run(self, ctx): # where would ctx come from ?!
        if ctx.iteration == 0:
            self.best_likelihood = np.inf
            self.early_stopping_count = 0
        if (ctx.iteration % 1) == 0:
            likelihood = - ctx.session.run(self.model.likelihood_tensor)
            self.logf.append(likelihood)
            #self.param_hist.append(copy.deepcopy(self.model.kern.variance.value))


            if likelihood < self.best_likelihood:
                self.best_likelihood = likelihood
                self.early_stopping_count = 0
            else:
                self.early_stopping_count += 1

        if (ctx.iteration % 1) == 0:
            print('Iteration: ', ctx.iteration, ' - Likelihood: ', likelihood, self.early_stopping_count)


adam = gpflow.train.AdamOptimizer(0.1).make_optimize_action(m)
logger = Logger(m)


print(m)
for i in range(20):
    loop = gpflow.actions.Loop([adam, logger], start=i*1, stop=(i+1)*1)()
    m.anchor(m.enquire_session())
    print(m)







mmin = np.min(train_features)
mmax = np.max(train_features)
pX = np.linspace(
    mmin + (mmax-mmin)*-1,
    mmax + (mmax-mmin)*1,
    100)[:, None]


'''
pX = np.linspace(
    self.svigp_parameters['plot_from'],
    self.svigp_parameters['plot_to'],
    self.svigp_parameters['n_grid'])[:, None]
'''
pY, pYv = m.predict_y(pX) # predicts mean and variance
plt.plot(train_features, train_labels, 'x')
line, = plt.plot(pX, pY, lw=1.5)
col = line.get_color()
plt.plot(pX, pY+2.*(pYv**0.5), col, lw=1.5) # mean +- 2 std
plt.plot(pX, pY-2.*(pYv**0.5), col, lw=1.5)

plt.plot(m.feature.Z.value, np.zeros(m.feature.Z.value.shape), 'k|', mew=2) # z values

# average line
plt.hlines(np.mean(train_labels), xmin=-2, xmax=2, color='red')

# stds
pstd = np.std(train_labels)
plt.hlines(np.mean(train_labels)+2*pstd, xmin=-2, xmax=2, color='purple')
plt.hlines(np.mean(train_labels)-2*pstd, xmin=-2, xmax=2, color='purple')

emav = 0.
emac = 100.
hl = 0.5**(1./(60*24*3))
emas = []
for y in train_labels:
    emav = emav*hl+y
    emac = emac*hl+1
    emas.append(emav/emac)

plt.plot(train_features, emas, color='green')
plt.show()
