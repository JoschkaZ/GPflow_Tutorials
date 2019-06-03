import numpy as np
from matplotlib import pyplot as plt

def summit(N):
    return (N+1)*N/2

entropies = []

for Nexp in range(15):

    N = 2**Nexp
    sum = summit(N)

    entropy = 0.
    for i in range(1,N):
        p = i/sum
        entropy -= p*np.log(p)

    print(N, sum, entropy)
    entropies.append(entropy)

plt.plot(entropies)
plt.show()

#%%
for i in range(1,10):
    print(i)


#%%
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
y_std = []

for Nexp in range(14):
    N = 2**Nexp

    maxs = []
    for trial in range(10000):

        r = np.random.normal(size=N)
        max = np.max(r)
        maxs.append(max)

    x.append(N)
    y.append(np.mean(maxs))
    y_std.append(np.std(maxs))


print(y)
print(y_std)
print('s')
plt.errorbar(x=x,y=y,yerr=y_std)
plt.plot(x=x,y=y_std)
plt.show()
