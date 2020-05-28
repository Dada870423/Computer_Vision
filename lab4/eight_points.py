import matplotlib.pyplot as plt
import numpy as np
import cv2

def eight_points(x1, x2, T1, T2):
    #Af =0
    A = []
    for i in range(x1.shape[1]):
        A.append((x2[0,i]*x1[0,i], x2[0,i]*x1[1,i], x2[0,i],
                  x2[1,i]*x1[0,i], x2[1,i]*x1[1,i], x2[1,i],
                  x1[0,i],       x1[1,i],       1))
    A = np.array(A, dtype='float')
    #solve SVD
    U, S, V = np.linalg.svd(A)
    f = V[:, -1]
    F = f.reshape(3,3).T
    
    #make det(F) = 0 
    U, D, V = np.linalg.svd(F)
    D[2] = 0
    S = np.diag(D)
    F = np.dot(np.dot(U, S), V)
    
    #De-normalize
    F = np.dot(np.dot(T2.T, F), T1)
    F = F/F[2,2] 
    return F
    
