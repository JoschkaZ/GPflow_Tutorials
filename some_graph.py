import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel




x = np.linspace(-3,5,100)
X = np.array([-2., -0.5, 0.8, 3, 4.5])

x_s = np.array([-1.2, 0.25, 1.5, 4])

def f(x):

    return np.sin(x+0.8)+x*0.2+1


print(x)
y = f(x)
Y = f(X)
print(Y)


kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X.reshape(-1,1), Y)

y_mean, y_cov = gp.predict(x[:, np.newaxis], return_cov=True)
y_s_mean, _ = gp.predict(x_s[:, np.newaxis], return_cov=True)

plt.rcParams.update({'font.size': 13})


plt.plot(x,y, '--', label='True Relationship', color='blue')
plt.plot(X,Y, '8', color='green', label='Maps used for Training',linewidth=4.0)
plt.xticks([])
plt.plot(x, y_mean, '--', label='Learned Relationship', color='orange')



plt.plot(x_s,y_s_mean, 'v', color='red', label='Generated Maps',linewidth=4.0)

plt.yticks([])
plt.xlabel('A Physics Parameter')
plt.ylabel('An Abstract Property of the Map')
plt.legend()
plt.savefig('somegraph.png', dpi=2000)
