import numpy as np
from numpy.linalg import inv
import cv2
import random

class RANSAC():
    def __init__(self, thresh, n_times, points):
        self.thresh = thresh
        self.n_times = n_times
        self.points = points


    def GeoDis(self, points, H):
        point1 = np.transpose(np.array([points[0], points[1], 1]))
        Estm = np.dot(H, point1)
        Estm2 = (1 / Estm.item(2)) * Estm

        point2 = np.transpose(np.array([points[2], points[3], 1]))
        Err = point2 - Estm2
        return np.linalg.norm(Err)
    
    def Cal8points_err(self, x1, x2, F):
        '''compute error of x F xp 
        '''
        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        test_err = (np.diag(np.dot(np.dot(x1.T, F), x2)))**2 / denom
        return test_err
    
    def Eight_points(self, x1, x2, T1, T2):
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
    
    def ransac_8points(self, x1, x2, T1, T2):
        '''
        input x, xp, T, Tp
        output F, inliers
        '''
        Ans_F = None
        max_inlier = []
        npts = x1.shape[1]
        
        for iter_1 in range(self.n_times):
            # spilt 8 points and other for test
            all_idxs = np.arange(npts)
            np.random.shuffle(all_idxs)
            try_idxs = all_idxs[:8]
            test_idxs = all_idxs[8:]
            try_x1 = x1[:, try_idxs]
            try_x2 = x2[:, try_idxs]
            test_x1 = x1[:, test_idxs]
            test_x2 = x2[:, test_idxs]
            
            #calculate possible F
            maybe_F = self.Eight_points(try_x1, try_x2, T1, T2)
            #maybe_F, _ = cv2.findFundamentalMat(try_x1.T, try_x2.T, cv2.FM_LMEDS)
            # get error of every pair for maybe_F
            test_err = self.Cal8points_err(test_x1, test_x2, maybe_F)
            #print(test_err.mean())
            now_inlier = list(try_idxs)
            for iter_err in range(len(test_err)):
                if test_err[iter_err] < self.thresh:
                    now_inlier.append(test_idxs[iter_err])
            
            if len(now_inlier) > len(max_inlier):
                Ans_F = maybe_F
                max_inlier = now_inlier
        
        if Ans_F is None:
            raise ValueError("didn't find F")
        
        return Ans_F, max_inlier

    def ransac(self, CorList, T1, T2):
        MaxLines = []
        AnsH = None
        Clen = len(CorList)
        for iter_1 in range(self.n_times):
            testP = []
            RanPoints = []
            ## pick up 4 random points
            Cor1 = CorList[random.randrange(0, Clen)]
            Cor2 = CorList[random.randrange(0, Clen)]
            Cor3 = CorList[random.randrange(0, Clen)]
            Cor4 = CorList[random.randrange(0, Clen)]
            Cor5 = CorList[random.randrange(0, Clen)]
            Cor6 = CorList[random.randrange(0, Clen)]
            Cor7 = CorList[random.randrange(0, Clen)]
            Cor8 = CorList[random.randrange(0, Clen)]
            RanPoints = np.vstack((Cor1, Cor2, Cor3, Cor4, Cor5, Cor6, Cor7, Cor8))

            ## cal F
            H = self.CalF(RanPoints)
            ## Cal line
            Lines = []
            for iter_2 in range(Clen):
                d = self.GeoDis(points = CorList[iter_2], H = H)
                if d < 5:
                    Lines.append(CorList[iter_2])

            if len(Lines) > len(MaxLines):
                MaxLines = Lines
                AnsH = H

        return AnsH, MaxLines
        
    def CalF(self, RanPoints):
        print(Ranpoints.shape)
        
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

    def CalH(self, RanPointsssssss):
        AsmList = []

        for iter_pts in RanPointsssssss:
            p1 = np.array([iter_pts.item(0), iter_pts.item(1), 1])
            p2 = np.array([iter_pts.item(2), iter_pts.item(3), 1])
            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0, p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            AsmList.append(a1)
            AsmList.append(a2)

        AsmMtx = np.matrix(AsmList)
        U, Sigma, Vt = np.linalg.svd(AsmMtx)

        pre_H = np.reshape(Vt[8], (3, 3))
        H = (1 / pre_H.item(8)) * pre_H
        return H










