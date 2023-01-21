import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from random import choices
import cv2

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

def display_support_time(T_support, line):
    t_support = T_support[T_support[:, 0] == line]
    plt.scatter(t_support[:, 2], t_support[:, 0])
    for i in range(t_support.shape[0]):
        plt.plot([t_support[i, 2], t_support[i, 2] + t_support[i, 3]], [t_support[i, 0], t_support[i, 0]])
    plt.grid()
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

def support_time_filter(data, neighboor_size, T_support):
    data_filtered = np.array(data)
    d = (data_filtered[:, 0] - data_filtered[-1, 0])**2 + (data_filtered[:, 1] - data_filtered[-1, 1])**2
    index_nbh = np.logical_and(d <= neighboor_size, d != 0)
    if np.any(index_nbh):
        dT_recent = np.min(np.abs(data_filtered[index_nbh][:, 3] - data_filtered[-1, 3]))
    else: 
        dT_recent = 0
    if dT_recent > T_support:
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

annots = loadmat('synthetic_square.mat')
data = annots['data']
# data = annots['events']
# data[:, 3] = (data[:, 3]).astype(float)
# data = data[:2000, :]

print(data)

data_chunk = []
last_data_time = 0
T_delay = 20

T_support_max = T_delay
T_support_min = 0
i = 0

k = 1
eps = 10**(-3)
Te_min, Te_max = find_min_max_time(data)
fe_min, fe_max = 1/Te_max, 1/Te_min
alpha_min, alpha_max = k/(np.log10(fe_max + eps)), k/np.log10(fe_min)

alpha_min = 0.14*np.log(10)
alpha_max = 0.3*np.log(10)
# print(np.log10(fe_max + eps), np.log10(fe_min))
# print(alpha_min, alpha_max)

# nbh_size = 7
optical_flow = []

time_op = 0

count_out = 0
count_in = 0

T_support_list = []

fe = 0


for i in range(data.shape[0]):
    if not(data[i, 3] - last_data_time > T_delay) :
        data_chunk.append(data[i, :])
        optical_flow.append([0, 0, 0])
        
    else: 
        start = time.time()

        data_chunk.append(data[i, :])
        data_chunk_np = np.array(data_chunk)
        
        data_chunk_np = refractory_filter(data_chunk_np, [20, 1])

        if data_chunk_np[-1, 3] - data_chunk_np[0, 3] > T_support_max:
            if data_chunk_np[-1, 3] - data_chunk_np[-2, 3] != 0:
                fe_curr = 1/(data_chunk_np[-1, 3] - data_chunk_np[-2, 3])
                fe = 0.2*fe + 0.8*fe_curr

            if fe != 1:
                alpha = k/np.log10(fe)

            
            if alpha > alpha_max : alpha_max, const_fe_max = alpha, fe
            if alpha < alpha_min : alpha_min, const_fe_min = alpha, fe

                
            T_support = (T_support_max - T_support_min)/(alpha_max - alpha_min)*(alpha - alpha_min) + T_support_min
            T_support_list.append([data_chunk_np[-1, 0], data_chunk_np[-1, 1], data_chunk_np[-1, 3], T_support])
            data_chunk_np = support_time_filter(data_chunk_np, 2*np.sqrt(2), T_support)
            

        vx_list = []
        vy_list = []
        support_time_list = []

        count_out += 1
        for nbh_size in [11, 13, 15]:            
            # find neighborhood
            nbh = find_neighborhood(data_chunk_np, nbh_size, T_delay)
            # print(nbh)
            # print()

            # calculate optical flow
            if nbh.shape[0] >= 3:
                count_in += 1
                vx_cur, vy_cur, support_time_cur = calculate_optical_flow(nbh)

                vx_list.append(vx_cur)
                vy_list.append(vy_cur)
                support_time_list.append(support_time_cur)

        

                
        if len(vx_list) == 0: vx = 0 
        else: vx = np.median(vx_list)
        if len(vy_list) == 0: vy = 0 
        else: vy = np.median(vy_list)
        if len(support_time_list) == 0: support_time = 0 
        else: support_time = np.median(support_time_list)

        # ax = plt.axes(projection='3d')
        # ax.scatter(nbh[:, 0], nbh[:, 1], nbh[:, 2])
        # ax.quiver(nbh[:, 0], nbh[:, 1], nbh[:, 2], vx_cur, vy_cur, support_time_cur, length = 1, normalize = True)
        # plt.show()

        optical_flow.append([vx, vy, support_time])

        last_data_time = data_chunk_np[0, 3]

        data_chunk = data_chunk_np.tolist()
        data_chunk.pop(0)

        time_op = (time_op + time.time() - start)/2


# print(alpha_max, alpha_min)
T_support_list = np.column_stack((data[:, 0], data[:, 1], data[:, 3], np.array(optical_flow)[:,2]))


# find the most repeated value of T_support_list[:, 0] in T_support_list
most_repeated = np.bincount(T_support_list[:, 3].astype(int)).argmax()

print(count_in, count_out, count_in/count_out)

# display_support_time(T_support_list, most_repeated)

# print(optical_flow)
# display_data3D(data, stop = 2000, optical_flow= np.array(optical_flow))

image = 255*np.ones((512, 512, 3))
for i in range(np.int16(np.max(data[:, 3]))):
    data_t = data[data[:, 3] == i]
    optical_flow_t = np.array(optical_flow)[data[:, 3] == i]
    if data_t.shape[0] != 0:
        for j in range(data_t.shape[0]):
            cv2.circle(image, (np.int16(data_t[j, 1]*4), np.int16(data_t[j, 0]*4)), 1, (0, 255, 0), -1)
            start_point = (np.int16(data_t[j, 1]*4), np.int16(data_t[j, 0]*4))
            end_point = (np.int16(data_t[j, 1] + optical_flow_t[j, 1])*4, np.int16(data_t[j, 0] + optical_flow_t[j, 0])*4)
            thickness = 2
            tipLength = 0.1
            cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), thickness, tipLength=tipLength)

        cv2.imshow("optical flow", image)
        cv2.waitKey(100)
    image = 255*np.ones((4*128, 4*128, 3))


    # data_t = data[data[:, 3] == i]
    # optical_flow_t = np.array(optical_flow)[data[:, 3] == i]
    # if data_t.shape[0] != 0:
    #     print(data_t[:, 0].shape, data_t[:, 1].shape)
    #     plt.scatter(data_t[:, 0], data_t[:, 1])
    #     if np.all(optical_flow != None):
    #         print(data_t[:, 0], data_t[:, 1], optical_flow_t[:, 0], optical_flow_t[:, 1])
    #         plt.quiver(data_t[:, 0], data_t[:, 1], optical_flow_t[:, 0], optical_flow_t[:, 1], scale = 1, scale_units = 'xy', color = 'r')
    #     plt.grid()
    #     plt.show()
    #     plt.pause(0.1)
    #     plt.close()
# ransac(data[:50, :], 10, 4/10, 1, method = "linear regression")