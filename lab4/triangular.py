import math
import numpy as np

def triangular(P2, x, xp):
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=float)
    
    p1 = P1[0, :]
    p2 = P1[1, :]
    p3 = P1[2, :]
    
    pp1 = P2[0, :]
    pp2 = P2[1, :]
    pp3 = P2[2, :]
    
    pointsX =[]
    for p, pp in zip(x, xp):
        u = p[0]
        v = p[1]
        up = pp[0]
        vp = pp[1]
        
        A = np.array([u*p3.T - p1.T,
                      v*p3.T - p2.T,
                      up*pp3.T - pp1.T,
                      vp*pp3.T - pp2.T])
        
        U, S, V = np.linalg.svd(A)
        X = V[:, -1]
        pointsX.append(X)
    pointsX = np.array(pointsX)
    
    for i in range(pointsX.shape[1]-1):
        pointsX[:,i] = pointsX[:,i] / pointsX[:,3]
    pointsX = pointsX[:,:3].T
    return(pointsX)

def count_p_front(points, m):
    '''
    m = [R|t]
    c = -Rt
    '''
    camera_c = np.dot(-m[:, 0:3], m[:, 3].T)
    for pt in points.T:
        print(pt)

def find_true_E(m1, m2, m3, m4, x, xp):
    pt1 = triangular(m1, x, xp)
    pt2 = triangular(m2, x, xp)
    pt3 = triangular(m3, x, xp)
    pt4 = triangular(m4, x, xp)
    
    count_p_front(pt1, m1)
    