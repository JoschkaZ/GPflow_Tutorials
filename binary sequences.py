import numpy as np
import sys
from operator import itemgetter

def get_not(S):
    out = []
    for x in S:
        out.append(int(not x))
    return out

def get_and(S,n):
    out = []
    for i in range(n,len(S)):
        out.append(int(S[i] and S[i-n]))
    return out

S_in = [1,1,0,0,1,1,0,0,1]
S_in = [1,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0]

S_table = []
S_table.append([S_in,0, "I"])
Algs = ["I"]

done = False
c = 0
while done == False:

    ind = 0
    done2 = False
    while done2 == False:
        done2 = True
        S = S_table[ind][0]
        S_cost = S_table[ind][1]
        S_alg = S_table[ind][2]

        if S_alg.split(',')[-1] != 'Not' and S_alg + ',Not' not in Algs:

            new_S = get_not(S)
            new_S_cost = S_cost + 1
            S_table.append([new_S,new_S_cost,S_alg+',Not'])
            Algs.append(S_alg+',Not')

        elif S_alg + ',And1' not in Algs:
            new_S = get_and(S,1)
            new_S_cost = S_cost + 1
            S_table.append([new_S,new_S_cost,S_alg+',And1'])
            Algs.append(S_alg+',And1')

        else:
            ind += 1
            done2 = False


        if np.mean(S_table[-1][0]) == 0 or np.mean(S_table[-1][0]) == 1:
            for s in S_table:
                print(s)
            sys.exit()

        S_table = sorted(S_table, key=itemgetter(1))





        if c > 1000:
            done = True
        c += 1



print(S_table)
