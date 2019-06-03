import scipy.stats as st

st.norm.ppf(0.4)


#%%

import numpy as np
import matplotlib.pyplot as plt



x = np.random.normal(0, 1., size=10000) ** 3
plt.hist(x, bins=1000)
plt.show()
