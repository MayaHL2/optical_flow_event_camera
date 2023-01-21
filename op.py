import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def display_data3D(data, start = 0, stop = 500, optical_flow = None):
    ax = plt.axes(projection='3d')
    data_p = data[data[:, 2] > 0]
    data_n = data[data[:, 2] < 0]
    ax.scatter3D(data_p[start:stop, 0], data_p[start:stop, 1], data_p[start:stop, 3], 'blue')
    ax.scatter3D(data_n[start:stop, 0], data_n[start:stop, 1], data_n[start:stop, 3], 'red')
    if np.all(optical_flow != None):
        ax.quiver(data[start:stop, 0], data[start:stop, 1], data[start:stop, 3], optical_flow[start:stop, 0], optical_flow[start:stop, 1], 0, length = 10, normalize = True)
    plt.show()


# annots = loadmat('data/cropped_tv2.mat')
annots = loadmat('data/synthetic_square.mat')
data = annots['data']
# data = annots['events']
# data[:, 3] = (data[:, 3]).astype(float)
# data = data[:2000, :]

data[data[:, 2] == -1, 2] = 0

data = np.int16(data)

ts = 20
t0 = 1

N = data.shape[0]

ep = 1
delta = 1000

ev = np.zeros((np.int16(data[:, 1].max()) + 1, np.int16(data[:, 0].max()) + 1, 2))
# flow = np.zeros((np.int16(data[:, 1].max()) + 1, np.int16(data[:, 0].max()) + 1, 2, 3))

optical_flow = []

for i in range(N):
    value = 0
    flag = 0
    x, y, p, t = data[i, :]
    p0 = 1 - p 

    if (t - ev[x, y, p] < ts) or (t - ev[x, y, p0] < t0):
        Tf = 20
        neigh_f = np.array([[x-1, y], [x+1, y], [x, y-1], [x, y+1]])
        t_neigh_f = np.min(ev[neigh_f[:, 0], neigh_f[:, 1], :])

        if t - t_neigh_f < Tf:
            flag = 1
            ev[x, y, p] = t

    if flag == 1:
        n = 5
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
                # Vérifier les shapes
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
display_data3D(data, optical_flow = optical_flow)