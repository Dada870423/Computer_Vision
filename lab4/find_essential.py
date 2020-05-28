import math
import numpy as np

'''K = np.array ([[ 5426.566895, 0.678017, 330.096680],
              [0.000000, 5423.133301, 648.950012],
              [0.000000, 0.000000 ,1.000000]])
F = np.array([[-9.93891308e-07, -1.25972426e-05 , 2.24159524e-03], 
              [1.38959982e-05  ,1.90372615e-07 , 6.67760806e-03],
              [-1.92357876e-03, -8.93169746e-03 , 1.00000000e+00]])
'''
def find_E(K, F):
    E = np.dot(np.dot(K.T, F), K)
    print(E)
    U, D, V = np.linalg.svd(E)
    e = (D[0] + D[1]) / 2
    D[0] = D[1] = e
    D[2] = 0
    E = np.dot(np.dot(U, np.diag(D)), V.T)
    U, D, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    R1 = np.dot(np.dot(U, W), V.T)
    R2 = np.dot(np.dot(U, W.T), V.T)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    Tx = np.dot(np.dot(U, Z), U.T)
    t = np.array ([[(Tx[2][1])], [(Tx[0][2])], [Tx[1][0]]])
    m1 = np.concatenate((R1, t), axis=1)
    m2 = np.concatenate((R1, -t), axis=1)
    m3 = np.concatenate((R2, t), axis=1)
    m4 = np.concatenate((R2, -t), axis=1)
    # 回傳3x4的matrix
    return m1, m2, m3 ,m4