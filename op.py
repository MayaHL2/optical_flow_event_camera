import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

def display_data3D(data, start = 0, stop = 500, optical_flow = None):
    ax = plt.axes(projection='3d')
    data_p = data[data[:, 2] > 0]
    data_n = data[data[:, 2] < 0]
    ax.scatter3D(data_p[start:stop, 0], data_p[start:stop, 1], data_p[start:stop, 3], 'blue')
    ax.scatter3D(data_n[start:stop, 0], data_n[start:stop, 1], data_n[start:stop, 3], 'red')
    if np.all(optical_flow != None):
        ax.quiver(data[start:stop, 0], data[start:stop, 1], data[start:stop, 3], optical_flow[start:stop, 0], optical_flow[start:stop, 1], 0, length = 10, normalize = True)
    plt.show()


annots = loadmat('data/cropped_tv2.mat')
data = annots['cropped']

data = data[:, np.array([0, 1, 3, 2])]

data =  data[:10000, :]

# annots = loadmat('data/synthetic_square.mat')
# data = annots['data']
# data = annots['events']
# data[:, 3] = (data[:, 3]).astype(float)
# data = data[:2000, :]

# data[data[:, 2] == -1, 2] = 0

# data = np.int16(data)

ts = 20
t0 = 1

n = 5

Tmax = 30000
Tmin = 100
Nsamples = 10
Log_min = 20/np.log10(10000000)
Log_max = 20/np.log10(10000)

N = data.shape[0]

ep = 1000000
delta = 100000

row = np.int16(data[:, 1].max()) + 1
column = np.int16(data[:, 0].max()) + 1
row, column = np.max([row, column]), np.max([row, column])
ev = np.zeros((row, column, 2))

# flow = np.zeros((np.int16(data[:, 1].max()) + 1, np.int16(data[:, 0].max()) + 1, 2, 3))

optical_flow = []

for i in range(N):
    value = 0
    flag = 0
    x, y, p, t = data[i, :]
    x = np.int16(x)
    y = np.int16(y)
    p = np.int16(p)
    p0 = 1 - p 
    # (t - ev[x, y, p] < ts) or (t - ev[x, y, p0] < t0)
    if True:
        timing = data[i:i+Nsamples, 3]
        t1 = timing[1:]/1000000
        t2 = timing[:-1]/1000000
        t_supp = t1-t2
        t_supp = t_supp[t_supp!=0] # to avoid infinities
        f = 1/t_supp
        fmean = np.sum(f)/Nsamples
        Log = 20/np.log10(fmean)
        Tf = (((Tmax-Tmin)/(Log_max-Log_min))*(Log-Log_min)+Tmin)
        # Tf = 70000
        neigh_f = np.array([[(x-1)%row, y%column], [(x+1)%row, y%column], [(x)%row, (y-1)%column], [(x)%row, (y+1)%column]])
        # print(neigh_f)
        t_neigh_f = np.min(ev[neigh_f[:, 0], neigh_f[:, 1], :])
        if t - t_neigh_f < Tf:
            flag = 1
            ev[x, y, p] = t

    if flag == 1:
        # take the n*n neighborhood
        neigh = np.array([[x-i, y-j] for i in range(n) for j in range(n)])
        # find the number of events that are != 0 in the neighborhood
        n_neigh = np.sum(np.logical_or(ev[neigh[:, 0], neigh[:, 1], 0] != 0 , ev[neigh[:, 0], neigh[:, 1], 1] != 0))
        if n_neigh > 3:
            neigh[np.zeros((n, n, 2)) == ev[neigh[:, 0], neigh[:, 1], :]]
            mu_x = np.mean(neigh[:, 0])
            mu_y = np.mean(neigh[:, 1])
            mu_t = np.mean(ev[neigh[:, 0], neigh[:, 1], :])
            neigh_normalized = np.column_stack((neigh - np.array([mu_x, mu_y]), np.max(ev[neigh[:, 0], neigh[:, 1], :], axis = 1) - mu_t))
            cov = np.cov(neigh_normalized)
            eig_val, eig_vec = np.linalg.eig(cov)
            V = eig_vec[np.argmin(eig_val)]
            if eig_vec[0, 0] < ep:
                # A vérifier si on prend le max ou les deux valeurs
                d = -(V[0]*neigh[:, 0] + V[1]*neigh[:, 1] + V[2]*np.max(ev[neigh[:, 0], neigh[:, 1], :], axis = 1))
                t_est = -(V[0]*x + V[1]*y + d)/V[2]
                if np.sum(np.abs(t_est - t)) < delta:
                    temp = -V[2]/(V[0]**2 + V[1]**2)
                    vx = V[0]*temp
                    vy = V[1]*temp
                    print(vx, vy)
                else:
                    vx = 0
                    vy = 0
            else:
                vx = 0
                vy = 0
        else:
            vx = 0
            vy = 0
        

        value = 1
        optical_flow.append([vx, vy])

    if value == 0:
        optical_flow.append([0, 0])

optical_flow = np.array(optical_flow)
print(data.shape, optical_flow.shape)
# display_data3D(data, optical_flow = optical_flow)


image = 255*np.ones((4*row, 4*column, 3))
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
    image = 255*np.ones((4*row, 4*column, 3))