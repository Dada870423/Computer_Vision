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





    def ransac(self, CorList):
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
            RanPoints.append((Cor1, Cor2, Cor3, Cor4))# = np.vstack((Cor1, Cor2, Cor3, Cor4))

            RanPoints = np.vstack((Cor1, Cor2, Cor3, Cor4))

            ## cal H
            H = self.CalH(RanPoints)
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










