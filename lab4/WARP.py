import numpy as np
from numpy.linalg import inv
import cv2
import random

class WARP():
    def warp(self, img, H, outputShape = None):
        outputShape = tuple(outputShape)
        NewImg = self.Realwarp(img, H, outputShape)
        return NewImg

    def Realwarp(self, img, H, outputShape):
        MP = self.CaPrTr(
            np.arange(outputShape[0], dtype = np.float32),
            np.arange(outputShape[1], dtype = np.float32)
        )
        MP = MP.reshape(outputShape[0], outputShape[1], 2)
        MP = self.perspectiveTransform(MP, inv(np.array(H)))
        return np.swapaxes(self.reMP(img, MP), 0, 1)

    def CaPrTr(self, *arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype = dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[i, ...] = a
        return arr.reshape(la, -1).T


    def perspectiveTransform(self, MP, M):
        t1 = ((M[0, 0] * MP[:, :, 0]) + (M[0, 1] * MP[:, :, 1]) + M[0, 2]) / (
                    (M[2, 0] * MP[:, :, 0]) + (M[2, 1] * MP[:, :, 1]) + M[2, 2])
        t2 = ((M[1, 0] * MP[:, :, 0]) + (M[1, 1] * MP[:, :, 1]) + M[1, 2]) / (
                    (M[2, 0] * MP[:, :, 0]) + (M[2, 1] * MP[:, :, 1]) + M[2, 2])
        return np.stack((t1, t2), axis = 2)


    def reMP(self, src, MP):
        RMP = np.zeros((MP.shape[0],MP.shape[1],src.shape[2]),dtype=np.uint8)
        height = MP.shape[0]
        width = MP.shape[1]
        print("Doing mapping and bilinear interpolation...\n")
        for i in range(height):
            for j in range(width):
                x = MP[i][j][1]
                y = MP[i][j][0]
                x1 = int(x)
                x2 = x1+1
                y1 = int(y)
                y2 = y1+1
                if x1 >= 0 and y1 >= 0 and x2 < src.shape[0] and y2 < src.shape[1]:
                    # Bilinear interpolation
                    RMP[i][j] = (src[x1][y1] * (x2 - x)*(y2 - y) + src[x1][y2] * (x2 - x) * (y - y1) + src[x2][y1] * (x - x1) * (y2 - y) + src[x2][y2] * (x - x1) * (y - y1)).astype(np.uint8)
        return RMP

