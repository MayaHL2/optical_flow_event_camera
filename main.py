from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time

annots = loadmat('data/stripes.mat')
data = annots['events']

start = time.time()

# Function to display data 
def display_data3D(data, start = 0, stop = 500, optical_flow = None):
    ax = plt.axes(projection='3d')
    data_p = data[data[:, 2] > 0]
    data_n = data[data[:, 2] < 0]
    ax.scatter3D(data_p[start:stop, 0], data_p[start:stop, 1], data_p[start:stop, 3], 'blue')
    ax.scatter3D(data_n[start:stop, 0], data_n[start:stop, 1], data_n[start:stop, 3], 'red')
    if np.all(optical_flow != None):
        ax.quiver(data[start:stop, 0], data[start:stop, 1], data[start:stop, 3], optical_flow[start:stop, 0], optical_flow[start:stop, 1], optical_flow[start:stop, 2], length = 10, normalize = True)
    plt.show()


def find_SAE(data, t, x_max, y_max):
    data_t = data[data[:, 3] < t]
    SAE = np.zeros((x_max, y_max))
    for i in range(data_t.shape[0]):
        if SAE[data_t[i, 0], data_t[i, 1]] < data_t[i, 3]:
            SAE[data_t[i, 0], data_t[i, 1]] = data_t[i, 3]
    return SAE

# Refractory filter
def refractory_filter(data, T_ref):
    data_filtered = np.array(data)
    i = 0
    while i < np.shape(data)[0]:
        k = [False]*data_filtered.shape[0]
        T = T_ref[0]*np.abs(data_filtered[:i, 2] + data_filtered[i, 2])/2 + T_ref[1]*np.abs(data_filtered[:i, 2] - data_filtered[i, 2])/2
        k[:i] = data_filtered[i, 3] - data_filtered[:i, 3] < T
        index_filter = np.where(np.logical_and(data_filtered[k, 0] == data_filtered[i, 0], np.any(data_filtered[k, 1] == data_filtered[i, 1])))[0]
        if len(index_filter) > 0:
            data_filtered[i, :] = [0, 0, 0, 0]
        i +=1
    return np.delete(data_filtered, data_filtered[:, 2] == 0, axis= 0)

def support_time_filter(data, T_support, neighboor_size):
    data_filtered = np.array(data)
    i = 0
    for i in range(1, data_filtered.shape[0]):
        k = [False]*data_filtered.shape[0]
        k[:i] = data_filtered[i, 3] - data_filtered[:i, 3] < T_support
        d = (data_filtered[k, 0] - data_filtered[i, 0])**2 + (data_filtered[k, 1] - data_filtered[i, 1])**2
        index_filter = np.where(np.logical_and(d <= neighboor_size, d != 0))[0]
        if len(index_filter) == 0:
            data_filtered[i, :] = [0, 0, 0, 0]
    return np.delete(data_filtered, data_filtered[:, 2] == 0, axis= 0)

def filter_outliers(nbh, n, eps = 0.1):
    d = -np.sum(n*nbh, axis = 1)
    t_est = - (np.sum(nbh[:, :2]*n[:2], axis = 1) + d)/n[2]
    nbh[t_est > (1-eps)*nbh.shape[0]**2/2]
    return nbh

def PCA(x):
    x = x - np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    _, eig_vec = np.linalg.eig(cov)
    return eig_vec

def calculate_optical_flow(nbh):
    eig_vec = PCA(nbh)
    n = eig_vec[2] 
    nbh = filter_outliers(nbh, n)
    if n[2]<0:
        n = -n
    temp = -n[2]/(n[0]*n[0] + n[1]*n[1])
    vx = temp*n[0]
    vy = temp*n[1]
    support_time = (n[0]*n[0] + n[1]*n[1])/n[2]
    return vx, vy, support_time

def optical_flow(data, nbh_size, t_optical_flow):
    optical_flow_data = np.zeros((data.shape[0], 3))
    for i in range(1, data.shape[0]):
        k = np.array([False]*data.shape[0])
        k[i:] = data[i:, 3] - data[i, 3] < t_optical_flow
        k[:i] = np.array([True]*i)
        d = np.array([False]*data.shape[0])
        d[k] = (data[k, 0] - data[i, 0])**2 + (data[k, 1] - data[i, 1])**2 + (data[k, 3] - data[i, 3])**2 < np.floor(nbh_size/2)*np.floor(nbh_size/2)*2 + t_optical_flow**2
        nbh = data[d]
        nbh = nbh[:, np.array([0, 1, 3])]
        if d.sum() > 3:
            vx, vy, support_time = calculate_optical_flow(nbh)
            optical_flow_data[i, :] = [vx, vy, support_time]
    return optical_flow_data
    
# data = data[np.logical_and(data[:, 3] > 500, data[:, 3] < 1000)]

# data_filtered = refractory_filter(data[:100, :], [20, 1])
# optical_flow_data = optical_flow(data_filtered, 9, 50)
# display_data3D(data_filtered, 0, 100, optical_flow_data)
# print(data[:100, :])

# Check if data happened at the same time 
def check_same_time(data):
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            if data[i, 3] == data[j, 3]:
                print(i, j)
