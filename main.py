from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time

annots = loadmat('data/stripes.mat')
data = annots['events']

start = time.time()

# Function to display data 
def display_data3D(data, start = 0, stop = 500):
    ax = plt.axes(projection='3d')
    datap = data[data[:, 2] > 0]
    datan = data[data[:, 2] < 0]
    ax.scatter3D(datap[start:stop, 0], datap[start:stop, 1], datap[start:stop, 3], 'blue')
    ax.scatter3D(datan[start:stop, 0], datan[start:stop, 1], datan[start:stop, 3], 'red')
    plt.show()

# Refractory filter
def refractory_filter(data, T_ref):
    data_filtered = np.array(data)
    i = 0
    while i < np.shape(data)[0]:
        k = np.zeros(data_filtered.shape[0])
        k = k==1
        T = T_ref[0]*np.abs(data_filtered[:i, 2] + data_filtered[i, 2])/2 + T_ref[1]*np.abs(data_filtered[:i, 2] - data_filtered[i, 2])/2
        k[:i] = data_filtered[i, 3] - data_filtered[:i, 3] < T
        index_filter = np.where(np.logical_and(data_filtered[k, 0] == data_filtered[i, 0], np.any(data_filtered[k, 1] == data_filtered[i, 1])))[0]
        if len(index_filter) > 0:
            data_filtered[i, :] = [0, 0, 0, 0]
        i +=1
    return np.delete(data_filtered, data_filtered[:, 2] == 0, axis= 0)


# data = np.array([[1, 2, 1, 1], [1, 2, 1, 20], [1, 2, 1, 21], [1, 2, 1, 80], [1, 2, 1, 101]])
data_filtered = refractory_filter(data[100:, :], [20, 1])

print(time.time() - start)