import sys
import numpy as np
import numpy.random as rnd
import time
import gpflow
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from mpl_toolkits import mplot3d
import pandas as pd
pd.set_option('display.expand_frame_repr', False)



class SVIGP():

    def __init__(self):

        # SETTINGS

        self.M = 10 # number of inducing points
        self.n_fake = 5000
        self.mini_batch = 100
        self.training_iterations = 5000
        self.fake_noise = 0.1

        self.fake_from = -1.0
        self.fake_to = 1.0
        self.plot_from = -2
        self.plot_to = 2

        self.n_grid = 40
        self.n_data_shown = 500

        self.ARD = True



        self.X = []
        self.Y = []
        self.Z = []
        self.Y_grid = None
        self.grid_A = None
        self.grid_B = None
        self.X_dim = None
        self.m = None
        self.kern = None
        self.likelihood = None

    def get_fake_data_1d(self):
        self.X = rnd.rand(self.n_fake, 1) *(self.fake_to-self.fake_from)+self.fake_from

        self.Y = (np.sin(self.X * 3*3.14) + 0.3*np.cos(self.X * 9*3.14) + 0.5 * np.sin(self.X * 7*3.14)
        + rnd.randn(self.n_fake, 1) * self.fake_noise)
        self.X_dim = 1

        self.grid_A = np.linspace(self.plot_from, self.plot_to, self.n_grid)
        self.Y_grid = np.sin(self.grid_A * 3*3.14) + 0.3*np.cos(self.grid_A * 9*3.14) + 0.5 * np.sin(self.grid_A * 7*3.14)


        print('fake data made with shapes:')
        print('X: ', self.X.shape)
        print('Y: ', self.Y.shape)

    def get_fake_data_2d(self):
        # get 2d feature values
        self.X = np.random.uniform(self.fake_from, self.fake_to, size=(self.n_fake,2))

        # get noisy function values
        y = []
        for p in self.X:



            #y.append(np.sin(p[0]*4.) + np.sin(p[1]*1.)) # FUNCTION HERE
            y.append(np.sin(p[0]*5.) + np.sin(p[1]*2000.)) # FUNCTION HERE



        dy = self.fake_noise + 0.0 * np.random.random(len(y)) #minimum noise + uniform [0:1] -> use as std
        noise = np.random.normal(0, dy)
        self.Y = np.reshape(np.array(y) + noise, newshape=(self.n_fake, 1))
        #self.Y = np.reshape(np.array(y), newshape=(self.n_fake, 1))

        # make 2d grid for plotting
        self.grid_A, self.grid_B = np.meshgrid(
            np.linspace(self.plot_from, self.plot_to, self.n_grid),
            np.linspace(self.plot_from, self.plot_to, self.n_grid)
        )



        #self.Y_grid = np.sin(self.grid_A*4.) + np.sin(self.grid_B*1.) # FUNCTION HERE
        self.Y_grid = np.sin(self.grid_A*5.) + np.sin(self.grid_B*2000.) # FUNCTION HERE


        self.X_grid = x = np.vstack((self.grid_A.flatten(), self.grid_B.flatten())).T


        print('fake data made with shapes:')
        print('X: ', self.X.shape)
        print('Y: ', self.Y.shape)
        print('Y_grid: ', self.Y_grid.shape)

    def plot_fake_data_1d(self):
        plt.plot(self.grid_A, self.Y_grid)
        plt.plot(self.X, self.Y, 'x')
        plt.show()

    def plot_fake_data_2d(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        col_black = matplotlib.colors.colorConverter.to_rgba('black', alpha=.1)

        # plot true function
        ax.plot_surface(self.grid_A, self.grid_B, self.Y_grid, rstride=1, cstride=1, edgecolor=col_black,
                        color=col_black, alpha = 0.1)

        # plot data points
        ax.scatter(self.X[:,0], self.X[:,1], self.Y, c='red', cmap='viridis', linewidth=0.5);

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.show()

    def build_model(self):
        self.kern = gpflow.kernels.RBF(
            input_dim = self.X.shape[1], # number of dimensions
            variance = 1., # initial value for varance parameter (magnitude)
            lengthscales = 1., # if 1:no ARD, if np.ones(input_dim): yes ARD)
            active_dims = None,  # list of length input_dim: which columns of X are used
            ARD = self.ARD) # if true: one lengthscale per dimension, if false: single lengthscale, otherwise infered

        self.likelihood = gpflow.likelihoods.Gaussian(
            variance = 1. #initial variance of ehh.... parameter distribution?
        )

        m = gpflow.models.SVGP(
            X = self.X, # data matrix (N x D) # dataholder object, if minibatch_size not not: Minibatch object (inherits from DataHolder)
            Y = self.Y, # data matrix (N x P)
            kern = self.kern, # a kernel object
            likelihood = self.likelihood, # a likelihood object
            feat = None, # no idea
            mean_function = None, # a gpflow object (maps X to Y (for what?))
            num_latent = None, # number of latent process to use (default: dim of Y)
            q_diag = False, # if true: covariance is approximated with diagonal matrix
            whiten = True, # if true: whitened representation of inducing points is used
            minibatch_size = self.mini_batch, # if not none: uses minibatches with given size
            Z = self.Z, # matrix of pseudo inputs (M x D)
            num_data = None, # total number of observations (needed if external mini batches are used)
            q_mu = None, # mean and cholesky of covariance of variational gaussian posterior is constructed. if not none: it is checked if they have consistent shapes. if not specified they are initialized based on num_inducing and q_diag
            q_sqrt = None # see above
        )

        self.m = m

    def get_inducing_points_random(self):
        idx = np.random.randint(0, self.X.shape[0], self.M)
        self.Z = self.X[idx]
        print('Shape of Inducing points')
        print('Z: ', self.Z.shape)

    def print_model_attributes(self):
        print('\n MODEL ATTRIBUTES \n')
        print('\n children: \n', self.m.children) #name and instances of variables
        print('\n descendants: \n', self.m.descendants) #full list of node descendants
        print('\n feeds: \n', self.m.feeds) # tensorflow feed dictionary for passing to tf.Session.run()
        print('\n full_name: \n', self.m.full_name) # name of model for backwards compatability
        print('\n graph: \n', self.m.graph) # tensorflow graph object
        print('\n initializable_feeds: \n', self.m.initializable_feeds) # feed dictionary used along with initializables list at the initialize function
        print('\n initializables: \n', self.m.initializables) # list of tensorflow tensors to be initialized
        print('\n name: \n', self.m.name) # name assigned to node at creation
        print('\n parent: \n', self.m.parent) # parent of this node (objects variables)
        print('\n pathname: \n', self.m.pathname) # recursinve representation parent path + name assigned to object by its parent
        print('\n root: \n', self.m.root) # toop of parental tree (objects variables)
        print('\n tf_name_scope: \n', self.m.tf_name_scope) # method for composing gpflows tree name scopes
        print('\n tf_pathname: \n', self.m.tf_pathname) # method for defining path name for tensor at build time

        print('\n data_holders: \n', self.m.data_holders) # generator object
        print('\n empty: \n', self.m.empty) # true or false
        #print('\n fixed: \n', self.m.fixed)
        print('\n index: \n', self.m.index) # some index number
        print('\n likelihood_tensor: \n', self.m.likelihood_tensor) # likelihood tensor object
        print('\n non_empty_params: \n', self.m.non_empty_params) # generator object
        print('\n objective: \n', self.m.objective) # objective tensor
        print('\n parameters: \n', self.m.parameters) # generator object
        print('\n params: \n', self.m.params) # generator object
        print('\n prior_tensor: \n', self.m.prior_tensor) # tensor of prior
        print('\n trainable: \n', self.m.trainable) # true or false
        print('\n trainable_parameters: \n', self.m.trainable_parameters) # generator object
        print('\n trainable_tensors: \n', self.m.trainable_tensors) # list of trainable tensors

    def print_kernel_attributes(self):
        print('\n KERNEL ATTRIBUTES \n')
        print('\n children: \n', self.kern.children) #name and instances of variables
        print('\n descendants: \n', self.kern.descendants) #full list of node descendants
        print('\n feeds: \n', self.kern.feeds) # tensorflow feed dictionary for passing to tf.Session.run()
        print('\n full_name: \n', self.kern.full_name) # name of model for backwards compatability
        print('\n graph: \n', self.kern.graph) # tensorflow graph object
        print('\n initializable_feeds: \n', self.kern.initializable_feeds) # feed dictionary used along with initializables list at the initialize function
        print('\n initializables: \n', self.kern.initializables) # list of tensorflow tensors to be initialized
        print('\n name: \n', self.kern.name) # name assigned to node at creation
        print('\n parent: \n', self.kern.parent) # parent of this node (objects variables)
        print('\n pathname: \n', self.kern.pathname) # recursinve representation parent path + name assigned to object by its parent
        print('\n root: \n', self.kern.root) # toop of parental tree (objects variables)
        print('\n tf_name_scope: \n', self.kern.tf_name_scope) # method for composing gpflows tree name scopes
        print('\n tf_pathname: \n', self.kern.tf_pathname) # method for defining path name for tensor at build time

        print('\n data_holders: \n', self.kern.data_holders) # generator object
        print('\n empty: \n', self.kern.empty) # true or false
        #print('\n fixed: \n', self.kern.fixed)
        print('\n index: \n', self.kern.index) # some index number
        print('\n parameters: \n', self.kern.parameters) # generator object
        print('\n params: \n', self.kern.params) # generator object
        print('\n prior_tensor: \n', self.kern.prior_tensor) # tensor of prior
        print('\n trainable: \n', self.kern.trainable) # true or false
        print('\n trainable_parameters: \n', self.kern.trainable_parameters) # generator object
        print('\n trainable_tensors: \n', self.kern.trainable_tensors) # list of trainable tensors

    def check_minibatch_ground_truth(self):

        ground_truth = self.m.compute_log_likelihood()

        self.m.X.set_batch_size(self.mini_batch)
        self.m.Y.set_batch_size(self.mini_batch)
        evals = [self.m.compute_log_likelihood() for _ in range(100)]

        plt.hist(evals, bins = 20)
        plt.axvline(ground_truth)
        plt.show()

    def test_minibatch_speedup(self):

        mbps = np.logspace(-2, 0, 10) # fractions of data used as batch size

        times = []
        objs = []

        for mbp in mbps:
            batch_size = int(len(self.X) * mbp)
            self.m.X.set_batch_size(batch_size)
            self.m.Y.set_batch_size(batch_size)

            start_time = time.time()
            objs.append([self.m.compute_log_likelihood() for _ in range(20)])
            times.append(time.time() - start_time)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(mbps, times, 'x-')
        ax1.set_xlabel("Minibatch proportion")
        ax1.set_ylabel("Time taken")

        ax2.plot(mbps, np.array(objs), 'kx')
        ax2.set_xlabel("Minibatch proportion")
        ax2.set_ylabel("ELBO estimates")
        plt.show()

    def plot_predictions_1d(self):

        pX = np.linspace(self.plot_from, self.plot_to, self.n_grid)[:, None]
        pY, pYv = self.m.predict_y(pX) # predicts mean and variance

        plt.plot(self.X, self.Y, 'x')

        line, = plt.plot(pX, pY, lw=1.5)
        col = line.get_color()
        plt.plot(pX, pY+2*pYv**0.5, col, lw=1.5) # mean +- 2 std
        plt.plot(pX, pY-2*pYv**0.5, col, lw=1.5)

        plt.plot(self.m.feature.Z.value, np.zeros(self.m.feature.Z.value.shape), 'k|', mew=2) # z values

        plt.show()

    def plot_predictions_2d(self):

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        pY, pYv = self.m.predict_y(self.X_grid) # predicts mean and variance

        print(self.X_grid.shape)
        print(pY.shape)
        print(pYv.shape)

        # data
        idx = np.random.randint(0, self.X.shape[0], self.n_data_shown)
        ax.scatter(self.X[idx][:,0], self.X[idx][:,1], self.Y[idx], c='red', cmap='viridis', linewidth=0.5)

        # true function
        ax.plot_surface(self.grid_A, self.grid_B, self.Y_grid, rstride=1, cstride=1, edgecolor='black',
                        color='black', alpha = 0.1)

        # + std
        ax.plot_surface(self.grid_A, self.grid_B, np.reshape(pY+1.96*pYv, newshape=(self.n_grid, self.n_grid)), rstride=1, cstride=1, edgecolor='red',
                        color='red', alpha = 0.1)

        # - std
        ax.plot_surface(self.grid_A, self.grid_B, np.reshape(pY-1.96*pYv, newshape=(self.n_grid, self.n_grid)), rstride=1, cstride=1, edgecolor='red',
                        color='red', alpha = 0.1)




        #make 2d grid
        #A,B = np.meshgrid(np.linspace(grid_from, grid_to, n_grid), np.linspace(grid_from, grid_to, n_grid))
        #y_grid = f_grid(self.grid_A, self.grid_B)
        #x = np.vstack((A.flatten(), B.flatten())).T

        plt.show()


    def set_batch_size(self, bs=-1):
        self.m.X.set_batch_size(bs)
        self.m.Y.set_batch_size(bs)

    def train(self):

        self.m.feature.trainable = False

        #logger = self.run_adam(self.m, gpflow.test_util.notebook_niter(1000)) # notebook_niter is probably not necessary
        logger = self.run_adam(self.m, self.training_iterations)
        # logger is returned to access its log variables

        plt.plot(-np.array(logger.logf)) #logf are the logged likelyhoods
        plt.xlabel('iteration')
        plt.ylabel('ELBO')
        plt.show()

    def run_adam(self, model, iterations):
        adam = gpflow.train.AdamOptimizer().make_optimize_action(model) # make AdamOptimizer object
        logger = Logger(model) # makge logger object
        actions = [adam, logger] #make a list of actions. all of them have a run function
        loop = gpflow.actions.Loop(actions, stop=iterations)() #execute the actions iterations times
        model.anchor(model.enquire_session()) # must be tensorflow session. trainable_parameters are anchored
        return logger

class Logger(gpflow.actions.Action): #inherits from Action class
    # action is a metaclass for wrapping functions in a container
    # action has a watcher used to measure time spent on execution
    def __init__(self, model):
        self.model = model
        self.logf = [] # logs the likelihood every 10 iterations

    def run(self, ctx): # where would ctx come from ?!
        if (ctx.iteration % 10) == 0:
            likelihood = - ctx.session.run(self.model.likelihood_tensor)
            print('Iteration: ', ctx.iteration, ' - Likelihood: ', likelihood)
            self.logf.append(likelihood)



if __name__ == '__main__':

    s = SVIGP()

    #s.get_fake_data_1d()
    #s.plot_fake_data_1d()

    s.get_fake_data_2d()
    s.plot_fake_data_2d()

    s.get_inducing_points_random()

    s.build_model()

    s.print_model_attributes()
    #s.print_kernel_attributes()

    s.check_minibatch_ground_truth()

    s.test_minibatch_speedup()

    s.plot_predictions_2d()
    #s.plot_predictions_1d()

    s.train()
    s.plot_predictions_2d()

    s.print_model_attributes()
    s.print_kernel_attributes()

    print('Model after training: ', self.m.as_pandas_table())
    # TODO try with and without ARD, try adding a useless feature, etc...
