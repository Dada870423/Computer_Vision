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
    #print(E)
    (xxx, yyy) = E.shape
    for iii in range(xxx):
        for jjj in range(yyy):
           pass
            ### print(E[iii][jjj], end = " ")
        ### print("", end = "\n")
    U, D, V = np.linalg.svd(E)
    e = (D[0] + D[1]) / 2
    D[0] = D[1] = e
    D[2] = 0
    E = np.dot(np.dot(U, np.diag(D)), V)
    U, D, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    R1 = np.dot(np.dot(U, W), V.T)
    R2 = np.dot(np.dot(U, W.T), V.T)
    if np.linalg.det(V) < 0:
        V = -V 
    if np.linalg.det(R2) < 0:
        U = -U
    U3 = U[:, -1]
    #Tx = np.dot(np.dot(U, Z), U.T)
    #t = np.array ([[(Tx[2][1])], [(Tx[0][2])], [Tx[1][0]]])
    m1 = np.vstack((R1.T, U3)).T
    m2 = np.vstack((R1.T, -U3)).T
    m3 = np.vstack((R2.T, U3)).T
    m4 = np.vstack((R2.T, -U3)).T
    # 回傳3x4的matrix
    return m1, m2, m3 ,m4
