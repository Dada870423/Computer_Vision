import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

class BFMATCH():
    def __init__(self, thresh):
        self.thresh = thresh
        self.match = []
        self.asm = []
        self.x = []
        self.xp = []
    def Best2Matches(self, des1, des2):
        idx1 = 0
        for p1 in des1:
            best_m = []
            temp0 = cv2.DMatch(idx1, 0, math.sqrt((p1 - des2[0]).T.dot(p1 - des2[0])))
            temp1 = cv2.DMatch(idx1, 1, math.sqrt((p1 - des2[1]).T.dot(p1 - des2[1])))
            if temp0.distance < temp1.distance:
                best_m.append(temp0)
                best_m.append(temp1)       
            else:
                best_m.append(temp1)
                best_m.append(temp0)

            idx2 = 0
            for p2 in des2:
                dis = math.sqrt((p1-p2).T.dot((p1-p2)))
                if dis < best_m[0].distance:
                    best_m[0].trainIdx = idx2
                    best_m[0].distance = dis
                elif dis < best_m[1].distance:
                    best_m[1].trainIdx = idx2
                    best_m[1].distance = dis
                idx2 = idx2 + 1
            idx1 = idx1 + 1
            self.match.append(best_m)   
        return self.match

    def CorresspondenceAcrossImages(self, kp1, kp2):
        for i, (m, n) in enumerate(self.match):
            if m.distance < self.thresh * n.distance:
                self.asm.append(m)
                self.x.append(kp1[m.queryIdx].pt)
                self.xp.append(kp2[m.trainIdx].pt)

        self.x = np.asarray(self.x)
        self.xp = np.asarray(self.xp)
        return self.x, self.xp




