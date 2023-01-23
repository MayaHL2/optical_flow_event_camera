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

data =  data[:400002, :]

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

row = np.int16(data[:, 0].max()) + 1
column = np.int16(data[:, 1].max()) + 1
# row, column = np.max([row, column]), np.max([row, column])
ev = np.zeros((row, column, 2))

# flow = np.zeros((np.int16(data[:, 1].max()) + 1, np.int16(data[:, 0].max()) + 1, 2, 3))

optical_flow = []
count = 0

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
        Tf = 7000000
        neigh_f = np.array([[x-1, y], [(x+1), y], [(x), (y-1)], [(x), (y+1)]])
        neigh_f = neigh_f[(neigh_f[:, 0] >= 0) & (neigh_f[:, 0] < row) & (neigh_f[:, 1] >= 0) & (neigh_f[:, 1] < column)]
        t_neigh_f = np.max(ev[neigh_f[:, 0], neigh_f[:, 1], :])
        # print(Tf)
        if t - t_neigh_f < Tf:
            flag = 1
            ev[x, y, p] = t

    if flag == 1:
        # take the n*n neighborhood
        neigh = np.array([[x-k, y-l] for k in range(-n//2+1, n//2+1) for l in range(-n//2+1, n//2+1)])
        # delete all neighbors that are out of the image
        neigh = neigh[(neigh[:, 0] >= 0) & (neigh[:, 0] < row) & (neigh[:, 1] >= 0) & (neigh[:, 1] < column)]
        # find the number of events that are != 0 in the neighborhood
        n_neigh = np.sum(np.logical_or(ev[neigh[:, 0], neigh[:, 1], 0] != 0 , ev[neigh[:, 0], neigh[:, 1], 1] != 0))
        if n_neigh > 3:
            logic = np.logical_not(ev[neigh[:, 0], neigh[:, 1], :] == np.zeros((neigh.shape[0], 2)))
            neigh = neigh[logic[:, 0] | logic[:, 1]]
            mu_x = np.mean(neigh[:, 0])
            mu_y = np.mean(neigh[:, 1])
            mu_t = np.mean(ev[neigh[:, 0], neigh[:, 1], :])
            neigh_normalized = np.column_stack((neigh - np.array([mu_x, mu_y]), np.max(ev[neigh[:, 0], neigh[:, 1], :], axis = 1) - mu_t))
            cov = np.cov(neigh_normalized)
            eig_val, eig_vec = np.linalg.eig(cov)
            V = eig_vec[np.argmin(eig_val)]
            if eig_val[0] < ep:
                # A vérifier si on prend le max ou les deux valeurs
                d = -(V[0]*neigh[:, 0] + V[1]*neigh[:, 1] + V[2]*np.max(ev[neigh[:, 0], neigh[:, 1], :], axis = 1))
                t_est = -(V[0]*x + V[1]*y + d)/V[2]
                if np.sum(np.abs(t_est - t)) < delta:
                    temp = -V[2]/(V[0]**2 + V[1]**2)
                    vx = V[0]*temp
                    vy = V[1]*temp
                    # print(vx, vy)
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
        count += 1

    if value == 0:
        optical_flow.append([0, 0])

optical_flow = np.array(optical_flow)
optical_flow_norm = np.linalg.norm(optical_flow, axis = 1)
optical_flow_norm[optical_flow_norm == 0] = 1
optical_flow = optical_flow/optical_flow_norm[:, None]


# display_data3D(data, optical_flow = optical_flow)
print(count)

print(optical_flow.max(), optical_flow.min(), optical_flow.mean(), optical_flow.std())

step = 10000
image = 255*np.ones((row, column, 3))
for i in range(step, 400002, step):
    # data_t = data[np.int16(data[:, 3]) == i]
    # print(data_t.shape[0], i, )
    # optical_flow_t = np.array(optical_flow)[data[:, 3] == i]
    data_t = data[i-step:i, :]
    optical_flow_t = np.array(optical_flow)[i-step:i, :]
    if data_t.shape[0] != 0:
        for j in range(data_t.shape[0]):
            # cv2.circle(image, (np.int16(data_t[j, 0]), np.int16(data_t[j, 1])), 1, (0, 255, 0), -1)
            start_point = (np.int16(data_t[j, 0].real), np.int16(data_t[j, 1].real))
            end_point = (np.int16(data_t[j, 0]+ 15*np.sign(optical_flow_t[j, 0].real)*np.abs(optical_flow_t[j, 0])), np.int16(data_t[j, 1] + 15*np.sign(optical_flow_t[j, 1].real)*np.abs(optical_flow_t[j, 1])))
            thickness = 1
            tipLength = 0.1
            cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), thickness, tipLength=tipLength)
    
    cv2.imshow("optical flow", image)
    cv2.waitKey(-1)
    image = 255*np.ones((row, column, 3))