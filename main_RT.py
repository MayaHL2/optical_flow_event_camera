import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from random import choices

# Function to display data 
def display_data3D(data, start = 0, stop = 500, optical_flow = None):
    ax = plt.axes(projection='3d')
    data_p = data[data[:, 2] > 0]
    data_n = data[data[:, 2] < 0]
    # ax.scatter3D(data_p[start:stop, 0], data_p[start:stop, 1], data_p[start:stop, 3], 'blue')
    # ax.scatter3D(data_n[start:stop, 0], data_n[start:stop, 1], data_n[start:stop, 3], 'red')
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
    min_time = 1000
    max_time = 0
    for i in range(1, N):
        time_diff = np.abs(events[i, 3] - events[i-1, 3])
        if time_diff < min_time and time_diff > 1:
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

def linear_regression(X, y , l_regularization = 0):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l_regularization*np.eye(X.shape[1])), X.T), y)

def gradient_descent(X, y, theta_0, alpha = 0.1, nbr_it = 10, l_regularization = 0):
  theta = theta_0
  for i in range(nbr_it):
    theta += alpha*1/(X.shape[0])*(np.dot(X.T,(y - np.dot(theta, X.T)))  + l_regularization*np.linalg.norm(theta))
  return theta

# Fit plane in 3D using RANSAC
def ransac(data, min_pts, in_over_out_ref, tol_err, method = "linear regression"):
    n_samples = data.shape[0]

    i_in = choices(range(n_samples), k = min_pts)
    x_i = data[:, 0][i_in]
    y_i = data[:, 1][i_in]
    t_i = data[:, 3][i_in]

    X_i = np.concatenate((x_i.reshape(-1, 1), y_i.reshape(-1, 1)), axis = 1)

    in_over_out = min_pts/n_samples
    

    while in_over_out < in_over_out_ref:
        print(in_over_out, in_over_out_ref)

        if method == "linear regression":
            theta_ransac = linear_regression(X_i, t_i)
        elif method == "gradient descent":
            theta_ransac = gradient_descent(X_i, t_i, np.zeros(X_i.shape))

        inliers = np.abs(data[:, 3] - np.dot(theta_ransac, data[:, :2].T)) < tol_err
        print(np.abs(data[:, 3] - np.dot(theta_ransac, data[:, :2].T))<tol_err)
        x_i = data[:, 0][inliers]
        y_i = data[:, 1][inliers]
        t_i = data[:, 3][inliers]

        X_i = np.concatenate((x_i.reshape(-1, 1), y_i.reshape(-1, 1)), axis = 1)

        in_over_out = t_i.shape[0]/n_samples

    return theta_ransac, X_i, y_i


def find_neighborhood(data, nbh_size, t_optical_flow):
    # The data for which we are looking for is the middle of the data array
    i = data.shape[0]//2
    # The neighborhood is within an ellipsoide 
    d = (data[:, 0] - data[i, 0])**2 + (data[:, 1] - data[i, 1])**2 < np.floor(nbh_size/2)*np.floor(nbh_size/2)*2
    d = np.logical_and((data[:, 3] - data[i, 3]) < t_optical_flow, d)
    nbh = data[d]
    return nbh[:, np.array([0, 1, 3])]

def filter_outliers(nbh, n, eps = 0.1, eps_n = 10**(-3)):
    d = -np.sum(n*nbh, axis = 1)
    t_est = - (np.sum(nbh[:, :2]*n[:2], axis = 1) + d)/(n[2] + eps_n)
    nbh[t_est > (1-eps)*nbh.shape[0]**2/2]
    return nbh

# PCA algorithm
def PCA(x):
    x = x - np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    return eig_vec[np.argmin(eig_val)]

def calculate_optical_flow(nbh):
    # Find the eigen vectors
    eig_vec = PCA(nbh)
    # The smallest eigen vector is the normal vector to the plane
    n = eig_vec
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
data[:, 3] = (data[:, 3]).astype(float)
data = data[:2000, :]

data_chunk = []
last_data_time = 0
T_delay = 300

T_support_max = T_delay
T_support_min = 0
i = 0

k = 1
eps = 10**(-3)
Te_min, Te_max = find_min_max_time(data)
fe_min, fe_max = 1/Te_max, 1/Te_min
alpha_min, alpha_max = k/(np.log(fe_max + eps)), k/np.log(fe_min)
# print(np.log(fe_max + eps), np.log(fe_min))
# print(alpha_min, alpha_max)

# nbh_size = 7
optical_flow = []

time_op = 0

count_out = 0
count_in = 0


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
            # if data_chunk_np[-1, 3] - data_chunk_np[-2, 3] == 0: 
            #     fe = 1
            # else: 
            fe = 1/(data_chunk_np[-1, 3] - data_chunk_np[-2, 3] +1) # J'ai ajouté le +1 pour éviter les divisions par 0 (au lieu de mettre un eps)
            alpha = k/np.log(fe + eps)
            T_support = (T_support_max - T_support_min)/(alpha_max - alpha_min)*(alpha - alpha_min) + T_support_min
            print(T_support)
            data_chunk_np = support_time_filter(data_chunk_np, T_support)


        vx_list = []
        vy_list = []
        support_time_list = []

        count_out += 1
        for nbh_size in [11, 13, 15]:            
            # find neighborhood
            nbh = find_neighborhood(data_chunk_np, nbh_size, T_delay)
            #print(nbh.shape[0])
            # calculate optical flow
            if nbh.shape[0] >= 3:
                count_in += 1
                vx_cur, vy_cur, support_time_cur = calculate_optical_flow(nbh)

                vx_list.append(vx_cur)
                vy_list.append(vy_cur)
                support_time_list.append(support_time_cur)


        vx = np.median(vx_list)
        vy = np.median(vy_list)
        support_time = np.median(support_time_list)
        optical_flow.append([data_chunk_np[data_chunk_np.shape[0]//2, :], vx, vy, support_time])

        last_data_time = data_chunk_np[0, 3]

        data_chunk = data_chunk_np.tolist()
        data_chunk.pop(0)

        time_op = (time_op + time.time() - start)/2


print(count_in, count_out)
optical_flow = np.array(optical_flow)
print(optical_flow[optical_flow[:, 1] !=0].shape)
print(optical_flow[optical_flow[:, 2] !=0].shape)
print(optical_flow[optical_flow[:, 3] !=0].shape)

display_data3D(data, stop = optical_flow.shape[0], optical_flow = optical_flow[:, 1:])

# ransac(data[:50, :], 10, 4/10, 1, method = "linear regression")