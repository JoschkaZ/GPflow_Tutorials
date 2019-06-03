
import random
import numpy as np

r = 0.99
c = 1
cs = []

for i in range(100000000):


    if random.random() < r:
        c += 1
    else:
        cs.append(c)
        c = 1

print(cs)
print(np.median(cs))

print(np.mean(cs))


#%%
import math
math.factorial(5)
#%%

print(1*(1-1)+2*1*(1-1/2)+3*1*1/2*(1-1/3)+4*1*1/2*1/3*(1-1/4))

sum = 0.
for i in range(100):
    N=i+1
    sum += N*(1.-1./N)*1./(math.factorial(N-1))

print(sum)

#%%
print(math.factorial(-1))



#%%


def get_AND(S):
    out = np.zeros((len(S),len(S)))
    out_cost = np.zeros((len(S),len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            out[i,j] = int(S[i] and S[j])
            out_cost[i,j] = max(i,j) + 1
    return out, out_cost

def get_I(S):

    out_cost = np.zeros(len(S))
    for i in range(len(S)):
        out_cost[i] = i
    return S, out_cost

def get_NOT(S):
    N = len(S)
    out = np.zeros(len(S))
    out_cost = np.zeros(len(S))
    for i in range(N):
        out[i] = int(not S[i])
        out_cost[i] = i+1
    return out, out_cost








# given a sequence S of length N, goal is to find N+1
# logical cost represents the number of actions required by a turing machine

import numpy as np
import matplotlib.pyplot as plt
S = [1,0,1,0,1,0]

subPs = [[S[0:n+1], S[n+1]] for n in range(len(S)-1)]
for subP in subPs:
    print(subP)
#%%
for subP in subPs:
    subS = subP[0]
    subA = subP[1]
    I, I_cost = get_I(subS)
    NOT, NOT_cost = get_NOT(subS)
    AND, AND_cost = get_AND(subS)


    print(' ')
    print(I)
    print(I_cost)
    print(' ')
    print(NOT)
    print(NOT_cost)
    print(' ')
    print(AND)
    print(AND_cost)
    print(' ')



#%%
N = len(S)

AND1 = np.zeros((len(S),len(S)))
AND1_costs = np.zeros((len(S),len(S)))
for i in range(len(S)):
    for j in range(len(S)):
        AND1[i,j] = S[i] and S[j]
        AND1_costs[i,j] = ((N-1)-i) + np.abs(i-j) + 1 # move to i, then move to j, then apply the and
plt.imshow(AND1, cmap="gray")
plt.show()
plt.imshow(AND1_costs, cmap="gray")
plt.show()
