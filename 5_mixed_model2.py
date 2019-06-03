import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import gpflow
from gpflow.test_util import notebook_niter, is_continuous_integration
from scipy.cluster.vq import kmeans2
from tensorflow.examples.tutorials.mnist import input_data

float_type = gpflow.settings.float_type
ITERATIONS = notebook_niter(1000) #how many training iterations are performed

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=False)

class Mnist:
    input_dim = 784
    Nclasses = 10
    X = mnist.train.images.astype(float)
    Y = mnist.train.labels.astype(float)[:, None]
    Xtest = mnist.test.images.astype(float)
    Ytest = mnist.test.labels.astype(float)[:, None]

def cnn_fn(x, output_dim):
    """
    Adapted from https://www.tensorflow.org/tutorials/layers
    """
    # input [BXYC]=[B,28,28,1]

    conv1 = tf.layers.conv2d(
          inputs=tf.reshape(x, [-1, 28, 28, 1]),
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    # [B, 28,28,32]

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # [B, 14,14,32]

    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    # [B, 14,14,64]


    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # [B, 7,7,64]

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # [B, 7*7*64]

    # -> [O]
    return tf.layers.dense(inputs=pool2_flat, units=output_dim, activation=tf.nn.relu)


gpflow.reset_default_graph_and_session()

minibatch_size = notebook_niter(1000, test_n=10) # TODO test_n ?
gp_dim = 5 # output dimension of tf model
M = notebook_niter(100, test_n=5) # number of inducing points TODO test_n ?

## placeholders
X = tf.placeholder(tf.float32, [minibatch_size, Mnist.input_dim])  # fixed shape so num_data works in SVGP
Y = tf.placeholder(tf.float32, [minibatch_size, 1])
Xtest = tf.placeholder(tf.float32, [None, Mnist.input_dim])

## build graph

with tf.variable_scope('cnn'):
    f_X = tf.cast(cnn_fn(X, gp_dim), dtype=float_type) # conv net is applied

with tf.variable_scope('cnn', reuse=True):
    f_Xtest = tf.cast(cnn_fn(Xtest, gp_dim), dtype=float_type)

gp_model = gpflow.models.SVGP(f_X, tf.cast(Y, dtype=float_type),
                              gpflow.kernels.RBF(gp_dim), gpflow.likelihoods.MultiClass(Mnist.Nclasses),
                              Z=np.zeros((M, gp_dim)), # we'll set this later
                              num_latent=Mnist.Nclasses)

loss = -gp_model.likelihood_tensor

m, v = gp_model._build_predict(f_Xtest)
my, yv = gp_model.likelihood.predict_mean_and_var(m, v)

with tf.variable_scope('adam'):
    opt_step = tf.train.AdamOptimizer(0.001).minimize(loss)

tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam')
tf_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn')

## initialize
sess = tf.Session()
sess.run(tf.variables_initializer(var_list=tf_vars))
gp_model.initialize(session=sess)

## reset inducing (they live in a different space as X, so need to be careful with this)
ind = np.random.choice(Mnist.X.shape[0], minibatch_size, replace=False)

fZ = sess.run(f_X, feed_dict={X:Mnist.X[ind]})
# Z_0 = kmeans2(fZ, M)[0] might fail
Z_0 = fZ[np.random.choice(len(fZ), M, replace=False)]

def set_gp_param(param, value):
    sess.run(tf.assign(param.unconstrained_tensor, param.transform.backward(value)))

set_gp_param(gp_model.feature.Z, Z_0)

## train
print('start training')
for i in range(ITERATIONS):
    if i % 10 == 0: print ('iteration: ', i)
    ind = np.random.choice(Mnist.X.shape[0], minibatch_size, replace=False)
    sess.run(opt_step, feed_dict={X:Mnist.X[ind], Y:Mnist.Y[ind]})

## predict
out_my, out_yv = sess.run((my, yv), feed_dict={Xtest:Mnist.Xtest})
print(out_my)
print(out_yv)
preds = np.argmax(sess.run(my, feed_dict={Xtest:Mnist.Xtest}), 1).reshape(Mnist.Ytest.shape)
correct = preds == Mnist.Ytest.astype(int)
acc = np.average(correct.astype(float)) * 100.
print('acc is {:.4f}'.format(acc))

gpflow.reset_default_graph_and_session()
