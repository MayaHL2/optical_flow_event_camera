import numpy as np

row, col = 240, 180
IM1 = np.zeros((row, col, 4), dtype=object)

def DVS_SG_parameters(neigh, order):
    idx = int(np.floor(neigh/2))
    A = np.zeros((neigh**2, int((order+1)*(order +2)/2)))
    a = np.zeros((int((order+1)*(order+2)/4), int((order+1)*(order + 2)/4)))
    ii = 1
    jj = 1
    for jjj in range(-idx, idx+1):
        for iii in range(-idx, idx+1):
            for j in range(order+1):
                for i in range(order-j+1):
                    A[ii-1, jj-1] = (iii**i)*(jjj**j)
                    jj += 1
            ii += 1
            jj = 1
    C = np.linalg.pinv(A)
    return C, a

def DVS_pass(event, v, sign, neigh):
    global im_v_pos
    global im_v_neg
    global row
    global col
    idx = 3
    flag = 1
    if event[3] > idx and event[3] < row-idx and event[2] > idx and event[2] < col-idx:
        if sign == 2:
            data = np.reshape(im_v_pos[event[3]-idx:event[3]+idx+1, event[2]-idx:event[2]+idx+1,:], 49, 3)
            data = data[data[:, 2] != 0]
            if data.size != 0:
                meanv = np.mean(data[:, 2])
                stdv = np.std(data[:, 2])
                lowerv = meanv - 0.5*stdv
                upperv = meanv + 0.5*stdv
                dir = np.arctan2(data[:, 1], data[:, 0])*180/np.pi + 180
                meandir = np.mean(dir)
                stddir = np.std(dir)
                lowerdir = meandir - 0.5*stddir
                upperdir = meandir + 0.5*stddir
            else:
                flag = 1
                return flag
            if v[2] < lowerv or v[2] > upperv or np.arctan2(v[1], v[0])*180/np.pi + 180 < lowerdir or np.arctan2(v[1], v[0])*180/np.pi + 180 > upperdir:
                flag = 0
                return flag

        elif sign == 3:
            data = np.reshape(im_v_neg[event[3]-idx:event[3]+idx+1, event[2]-idx:event[2]+idx+1,:], neigh**2, 3)
            data = data[data[:, 2] != 0]
            if data.size != 0:
                meanv = np.mean(data[:, 2])
                stdv = np.std(data[:, 2])
                lowerv = meanv - 0.5*stdv
                upperv = meanv + 0.5*stdv
                dir = np.arctan2(data[:, 1], data[:, 0])*180/np.pi + 180
                meandir = np.mean(dir)
                stddir = np.std(dir)
                lowerdir = meandir - 0.5*stddir
                upperdir = meandir + 0.5*stddir
            else:
                flag = 1
                return flag
            if v[2] < lowerv or v[2] > upperv or np.arctan2(v[1], v[0])*180/np.pi + 180 < lowerdir or np.arctan2(v[1], v[0])*180/np.pi + 180 > upperdir:
                flag = 0
                return flag
    else:
        flag = 0


def DVS_neighborhood(event, neigh, Sign, t0, dt):
    # n = {}
    # n['data'] = []
    # global IM1
    # global row 
    # global col
    idx = int(np.floor(neigh/2))
    if event[3] > idx and event[3] < row-idx and event[2] > idx and event[2] < col-idx:
        if not IM1[event[3]+1,event[2]+1,Sign]['time']:
            tc = IM1[event[3]+1,event[2]+1,Sign]['time'][-1]
        else:
            tc = 1e20
        ii = 1
        for i in range(-idx, idx+1):
            for j in range(-idx, idx+1):
                if not IM1[event[3]+i+1,event[2]+j+1,Sign]['time']:
                    if tc - IM1[event[3]+i+1,event[2]+j+1,Sign]['time'][-1] < dt:
                        n['data'].append([event[3]+ i , event[2]+j , (IM1[event[3]+i+1,event[2]+j+1,Sign]['time'][-1] - t0)*1e-6, 1])
                        ii = ii+1
    return n