import numpy as np
import matplotlib.pyplot as plt

preds = np.array([0.3,0.1, 0.7])
sig = np.array([0.01,0.01,500])
poss = [1,100,200]
korr = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        korr[i][j] = np.e**(-(poss[i]-poss[j])**2)

plt.imshow(korr, cmap="gray")
plt.show()

sigm = np.identity(len(sig)) * sig
print(sigm)
print(korr+sigm)
print(np.linalg.inv(korr+sigm))
print(np.dot(np.linalg.inv(korr+sigm),preds))

#%%

jitter = 1e-5
y = np.array([0,5,10,8,1,2,3]) # opinions
x = np.array([1,1.1,100,50,0,0,0]) # where the opinions sit in people-space
y = np.array([1,2,20])
x = np.array([1,1,10])



y.sort()
x.sort()
# people close to another are correlated / there opinions are correlated

plt.scatter(x,y)




K = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j] = np.e**(-(x[i]-x[j])**2) + (i==j)*jitter



#print(np.linalg.inv(K))
#print(np.dot(np.ones(3),K))
#print(1/(np.dot(np.ones(3),K)))
answer = (np.sum(y*1/(np.dot(np.ones(len(x)),K)) / np.sum(1/(np.dot(np.ones(len(x)),K)))))

plt.hlines(answer, min(x), max(x))
plt.show()
plt.imshow(K, cmap="gray")
plt.show()






#%%
import numpy as np
import scipy.stats

jitter = 1e-5

ym = np.array([2,2.05,9,10,5])
ys = np.array([1.5,1.49,0.5,0.7,0.25])
x = np.array([1,1.5,7,6.5,10])
ym = np.array([3,3.2,8])
ys = np.array([1,1,1])
x = np.array([1,2,10])

g = np.linspace(min(x), max(x), 200)



#plt.scatter(x,y)
gaussians = []
for i in range(len(ym)):
    gaussian = scipy.stats.multivariate_normal.pdf(g, ym[i], ys[i])
    plt.plot(g,gaussian /np.sum(gaussian))
    gaussians.append(gaussian)
gaussians = np.array(gaussians)

K = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j] = np.e**(-((x[i]-x[j])/0.1)**2) + (i==j)*jitter

#print(np.linalg.inv(K))
#print(np.dot(np.ones(3),K))
#print(1/(np.dot(np.ones(3),K)))
answer = 1/(np.dot(np.ones(len(x)),K)) / np.sum(1/(np.dot(np.ones(len(x)),K)))

print(gaussians)
gaussians = np.log(gaussians)
for i in range(len(answer)):
    gaussians[i] = gaussians[i]*answer[i]

print(gaussians.shape)
gaussians = np.sum(gaussians, axis = 0)
print(gaussians.shape)
gaussians = np.exp(gaussians)
print(gaussians.shape)
plt.plot(g,gaussians /np.sum(gaussians),color='black')
plt.show()
#print(gaussians)
plt.imshow(K, cmap="gray")
plt.show()
#%%
plt.hlines(answer, min(x), max(x))
plt.show()
plt.imshow(K, cmap="gray")
plt.show()
