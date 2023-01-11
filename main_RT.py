import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

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

# Refractory filter
def refractory_filter(data, T_ref):
    data_filtered = np.array(data)
    i = 0
    for i in range(data.shape[0]):
        # Create array of booleans : False : not close to current data, True : close to current data
        k = [False]*data.shape[0]
        # Compute the time threshold for each data (if p(i)+p(i-1) = 0 (<=> sign(p(i)) != sign(p(i-1)))  then T = T_ref[0], else T = T_ref[1])
        T = T_ref[0]*np.abs(data_filtered[:i, 2] + data_filtered[i, 2])/2 + T_ref[1]*np.abs(data_filtered[:i, 2] - data_filtered[i, 2])/2
        # Apply threshold on k array
        k[:i] = data_filtered[i, 3] - data_filtered[:i, 3] < T
        # Find the index of the data that are close to the current data
        index_filter = np.where(np.logical_and(data_filtered[k, 0] == data_filtered[i, 0], np.any(data_filtered[k, 1] == data_filtered[i, 1])))[0]
        # If there are data within the time scope T, then remove the current data
        if len(index_filter) > 0:
            data_filtered[i, :] = [0, 0, 0, 0]
    return np.delete(data_filtered, data_filtered[:, 2] == 0, axis= 0)

# find the highest and lowest time diffrence bewteen two events
def find_min_max_time(events):
    N = len(events)
    min_time = float('inf')
    max_time = float('-inf')
    for i in range(1, N):
        time_diff = events[i, 3] - events[i-1, 3]
        if time_diff < min_time and time_diff>0:
            min_time = time_diff
        if time_diff > max_time:
            max_time = time_diff
    return min_time, max_time

def support_time_filter(data, neighboor_size):
    data_filtered = np.array(data)
    d = (data_filtered[:, 0] - data_filtered[-1, 0])**2 + (data_filtered[:, 1] - data_filtered[-1, 1])**2
    index_filter = np.where(np.logical_and(d <= neighboor_size, d != 0))[0]
    if len(index_filter) == 0:
        data_filtered[-1, :] = [0, 0, 0, 0]
    return np.delete(data_filtered, data_filtered[:, 2] == 0, axis= 0)

# PCA algorithm
def PCA(x):
    x = x - np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    _, eig_vec = np.linalg.eig(cov)
    return eig_vec

def find_neighborhood(data, nbh_size, t_optical_flow):
    # The data for which we are looking for is the middle of the data array
    i = data.shape[0]//2
    # The neighborhood is within an ellipsoide 
    d = (data[:, 0] - data[i, 0])**2 + (data[:, 1] - data[i, 1])**2 + (data[:, 3] - data[i, 3])**2 < np.floor(nbh_size/2)*np.floor(nbh_size/2)*2 + t_optical_flow**2
    nbh = data[d]
    return nbh[:, np.array([0, 1, 3])]

def filter_outliers(nbh, n, eps = 0.1, eps_n = 10**(-3)):
    d = -np.sum(n*nbh, axis = 1)
    t_est = - (np.sum(nbh[:, :2]*n[:2], axis = 1) + d)/(n[2] + eps_n)
    nbh[t_est > (1-eps)*nbh.shape[0]**2/2]
    return nbh

def calculate_optical_flow(nbh):
    # Find the eigen vectors
    eig_vec = PCA(nbh)
    # The smallest eigen vector is the normal vector to the plane
    n = eig_vec[2] 
    # Filter outliers
    nbh = filter_outliers(nbh, n, eps = 10**(-3))
    # Change sign of n if the support time is negative
    if n[2] < 0:
        n = -n
    temp = -n[2]/(n[0]*n[0] + n[1]*n[1] + eps)
    # Calculate vx, vy and support time
    vx = temp*n[0]
    vy = temp*n[1]
    support_time = (n[0]*n[0] + n[1]*n[1])/(n[2] + eps)
    return vx, vy, support_time

annots = loadmat('data/stripes.mat')
data = annots['events']

data_chunk = []
last_data_time = 0
T_delay = 200

T_support_max = T_delay
T_support_min = 0
i = 0

k = 1
eps = 10**(-3)
Te_min, Te_max = find_min_max_time(data)
fe_min, fe_max = 1/Te_max, 1/Te_min
alpha_min, alpha_max = k/(np.log(fe_max + eps)), k/np.log(fe_min)

nbh_size = 7
optical_flow = []

time_op = 0


for i in range(data.shape[0]):
    if not(data[i, 3] - last_data_time > T_delay) :
            data_chunk.append(data[i, :])
            # size_chunk = len(data_chunk) + 1 
    else: 
        start = time.time()

        data_chunk.append(data[i, :])
        data_chunk_np = np.array(data_chunk)
        
        data_chunk_np = refractory_filter(data_chunk_np, [20, 1])

        if data_chunk_np[-1, 3] - data_chunk_np[0, 3] > T_support_max:
            fe = 1/(data_chunk_np[-1, 3] - data_chunk_np[-2, 3] + 1) # J'ai ajouté le +1 pour éviter les divisions par 0 (au lieu de mettre un eps)
            alpha = k/np.log(fe + eps)
            T_support = (T_support_max - T_support_min)/(alpha_max - alpha_min)*(alpha - alpha_min) + T_support_min
            data_chunk_np = support_time_filter(data_chunk_np, T_support)

        vx_prev = 0
        vy_prev = 0
        support_time_prev = 0
        for nbh_size in [7, 9, 11]:
            # find neighborhood
            nbh = find_neighborhood(data_chunk_np, nbh_size, T_delay)
            # print(nbh.shape[0])
            # calculate optical flow
            if nbh.shape[0] >= 3:
                vx_cur, vy_cur, support_time_cur = calculate_optical_flow(nbh)
                vx = (vx_cur + vx_prev)/2
                vy = (vy_cur + vy_prev)/2
                support_time = (support_time_cur + support_time_prev)/2

                print("curr", vx_cur, vy_cur, support_time_cur)

                vx_prev = vx_cur
                vy_prev = vy_cur
                support_time_prev = support_time_cur

        optical_flow.append([data_chunk_np[data_chunk_np.shape[0]//2, :], vx, vy, support_time])
        print(vx, vy, support_time)
        # vx_cur and vx are the same
        # because T_delay influences way more the data

        last_data_time = data_chunk_np[0, 3]

        data_chunk = data_chunk_np.tolist()
        data_chunk.pop(0)

        time_op = (time_op + time.time() - start)/2


print(time_op)
optical_flow = np.array(optical_flow)
print(optical_flow.shape)
display_data3D(data, stop = optical_flow.shape[0], optical_flow = optical_flow[:, 1:])