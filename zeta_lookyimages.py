
import pickle as pkl
import matplotlib.pyplot as plt

data = pkl.load(open(r"C:\\Outputs\\slices2_128_zeta_small.pkl", "rb"))


#%%

selection = []
for element in data:
    print(element[1])
    seed = int(element[1][4:].split('_')[0])
    zeta = float(element[1].split('ZETA_X')[1].split('_')[0])
    #print(seed)
    #print(zeta)

    if seed == 0 or 1==2:
        selection.append([zeta, element[0]])


print(selection[0:3])
print(len(selection))

#%%
hist_data = {}
for s in selection:

    plt.imshow(s[1])
    plt.title(s[0])
    plt.show()

    if s[0] not in hist_data:
        hist_data[s[0]] = []
    hist_data[s[0]]  += list(s[1].flatten())


#%%
for zeta in hist_data:
    print(zeta)
    plt.hist(hist_data[zeta], alpha=0.1, bins=100)
    plt.ylim((0,10000))
    plt.show()


#%%


[1,2,34]+[1,2,3]


#%%



import pickle as pkl
import matplotlib.pyplot as plt

data = pkl.load(open(r"C:\\Outputs\\slices2_128.pkl", "rb"))

for element in data:

    seed = int(element[1][4:].split('_')[0])
    zeta = float(element[1].split('ZETA_X')[1].split('_')[0])
    print(seed, zeta)
    plt.imshow(element[0])
    plt.show()
