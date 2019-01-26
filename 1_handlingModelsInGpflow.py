
# model class allows controlling parameters
# how to view models and parameters
# how to constrain parameters
# how to fix model parameters
# how to apply priors to paraemters
# how to optimize models

# build simple logistig regression model



import gpflow
import numpy as np

# simple GPR model without building it in TensorFlow graph
with gpflow.defer_build():
    X = np.random.rand(20, 1)
    print(X.shape)
    Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(20,1) * 0.01
    m = gpflow.models.GPR(X, Y, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.Linear(1))

    print('X: \n', X)
    print('\nY: \n', Y)
    print('\nModel: \n', m)
    # -> there are 4 parameters:
    # RBF kernel has varianve and lengthscale
    # linear kernel has variance
    # there is also variance of noise as part of the likelyhood
    print('\nLikelyhood parameters: \n', m.likelihood)

    # read trainables. currently all parameters constrained +ve. unconstrained representation: alpha = log(exp(theta)-1)
    print('\nReadTrainables: \n', m.read_trainables())
    # assign values

    m.kern.kernels[0].lengthscales = 0.5
    m.likelihood.variance = 0.01
    print('\nAfter changing parameters: \n', m)

    # constraints are handled by Transform classes
    # must be before compiling the model
    # change log(exp(theta)-1) to alpha = log(theta)
    m.kern.kernels[0].lengthscales.transform = gpflow.transforms.Exp()
    print('\nTrainables after changing transformation\n', m.read_trainables())

    # fix a parameters
    m.kern.kernels[1].variance.trainable = False
    print('\nAfter fixing a parameter\n', m)

    # priors are set using members of gpflow.priors
    m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2,3) #TODO what is Gamma?
    print('\nAfter setting a prior\n', m)

    # optimization is done by creating an instance of optimizer
    # minimise negative log-likelyhood
    # variables with priors are MAP-estimated, others are ML TODO what?
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print('\nAfter fitting\n', m)




# building new models

#need to inherrit from gpflow.models.Model
#parameters are instatiated with gpflow.Param
#gpflow.params.Parameterized acts as a container of Param

# implement linear multiclass classification
# using 2 parameters: weight matrix and bias
# will implement the private _built_likelihood method -> returns tf scalar representing the (log) likelyhood
# param objects will be inside _build_likelihood as unconstrained tensors

import tensorflow as tf
class LinearMulticlass(gpflow.models.Model):

    def __init__(self, X, Y, name=None):
        super().__init__(name=name) #call parent constructor

        self.X = X.copy() #numpy array of inputs
        self.Y = Y.copy() #1-of-k representation of labels

        self.num_data, self.input_dim = X.shape
        _, self.num_classes = Y.shape

        # make some funky parameters
        self.W = gpflow.Param(np.random.randn(self.input_dim, self.num_classes))
        # this is n_features x n_classes initiated uniform randomly ?
        self.b = gpflow.Param(np.random.randn(self.num_classes))

        # parameters must be attributed of the class!

    @gpflow.params_as_tensors
    def _build_likelihood(self): #no arguments
        p = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b) #param variables are tensorflow arrays
        # p is a vector with the probabilities for the classes
        return tf.reduce_sum(tf.log(p) * self.Y) # must return a scalar

# make up some data
X = np.vstack([np.random.randn(10,2) + [2,2],
               np.random.randn(10,2) + [-2,2],
               np.random.randn(10,2) + [2,-2]])
Y = np.repeat(np.eye(3), 10, 0)
print('\nX\n', X)
print('\nY\n', Y)

from matplotlib import pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12,6)
plt.scatter(X[:,0], X[:,1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)
plt.show()


m = LinearMulticlass(X, Y)
print('\nLinearMulticlass Model\n', m)

# train!
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)
m.as_pandas_table()

print('\nm after optimization\n', m)



xx, yy = np.mgrid[-4:4:200j, -4:4:200j]
X_test = np.vstack([xx.flatten(), yy.flatten()]).T #test points
f_test = np.dot(X_test, m.W.read_value()) + m.b.read_value() # predictions
p_test = np.exp(f_test) # mhmm why is stuff logged?
p_test /= p_test.sum(1)[:,None] #normalise the 3 probabilities

for i in range(3):
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.5], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.2], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.1], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.05], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.02], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.002], colors='k', linewidths=1)
    plt.contour(xx, yy, p_test[:,i].reshape(200,200), [0.0002], colors='k', linewidths=1)
plt.scatter(X[:,0], X[:,1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)
plt.show()
