import sys
import numpy as np
import numpy.random as rnd
import time
import gpflow
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

M = 10


def func(x):
    return np.sin(x * 3*3.14) + 0.3*np.cos(x * 9*3.14) + 0.5 * np.sin(x * 7*3.14)
X = rnd.rand(100, 1) * 2 - 1
Y = func(X) + rnd.randn(100, 1) * 0.2 # function + noise
plt.plot(X, Y, 'x')
plt.show()

D = X.shape[1]
Xt = np.linspace(-1.1, 1.1, 100)[:, None]
Yt = func(Xt)#




def init(): #model initialization
    kern = gpflow.kernels.RBF(D, 1) #D is number of features ? (just 1)
    Z = X[:M, :].copy() #select first M elements of X
    m = gpflow.models.SVGP(X, Y, kern, gpflow.likelihoods.Gaussian(), Z, minibatch_size=len(X))
    return m


m = init()
print(m)


ground_truth = m.compute_log_likelihood()
print('\nground truth:\n', ground_truth)
m.X.set_batch_size(10)
m.Y.set_batch_size(10)
print(m)

# get the log likelyhood for 100 randomly selected batches and plot in histogram
# the gp is not fitted yet so the number is very low...
evals = [m.compute_log_likelihood() for _ in range(100)]
print(evals)

plt.hist(evals)
plt.axvline(ground_truth)
plt.show()

'''
mbps = np.logspace(-2, 0, 10)
times = []
objs = []
for mbp in mbps:
    m.X.set_batch_size = int(len(X) * mbp)
    m.Y.minibatch_size = int(len(X) * mbp)
    start_time = time.time()
    objs.append([m.compute_log_likelihood() for _ in range(20)])
#    plt.hist(objs, bins = 100)
#    plt.axvline(ground_truth, color='r')
    times.append(time.time() - start_time)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(mbps, times, 'x-')
ax1.set_xlabel("Minibatch proportion")
ax1.set_ylabel("Time taken")

ax2.plot(mbps, np.array(objs), 'kx')
ax2.set_xlabel("Minibatch proportion")
ax2.set_ylabel("ELBO estimates")
plt.show()
'''


#%%



def plot():
    pX = np.linspace(-2, 2, 1000)[:, None]
    pY, pYv = m.predict_y(pX)
    plt.plot(X, Y, 'x')
    line, = plt.plot(pX, pY, lw=1.5)
    col = line.get_color()
    plt.plot(pX, pY+2*pYv**0.5, col, lw=1.5)
    plt.plot(pX, pY-2*pYv**0.5, col, lw=1.5)
    plt.plot(m.feature.Z.value, np.zeros(m.feature.Z.value.shape), 'k|', mew=2)

plot()
plt.title("Predictions before training")
plt.show()

st = time.time()
logt = []
logx = []
logf = []
def logger(x):
    if (logger.i % 10) == 0:
        logx.append(x)
        logf.append(m._objective(x)[0])
        logt.append(time.time() - st)
    logger.i+=1
logger.i = 1


m.X.set_batch_size(100)
m.Y.set_batch_size(100)

m.feature.trainable = False
opt = gpflow.train.AdamOptimizer()
#m.minimize(method=tf.train.AdamOptimizer(), maxiter=np.inf, callback=logger)
opt.minimize(m, maxiter=1000)


plot()
plt.title("Predictions after training")
plt.show()
