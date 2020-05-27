import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def draw(l, lp, x, xp, img1, img2):
    pic1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h, w = img1.shape

    for r, pt_x, pt_xp in zip(l, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0 = 0
        y0 = (-r[2]/r[1]).astype(np.int)
        x1 = w
        y1 = (-(r[2]+r[0]*w)/r[0]).astype(np.int)
        pic1 = cv2.line(pic1, (y0,x0), (y1,x1), color, 1)
        pic1 = cv2.circle(pic1, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        pic2 = cv2.circle(pic2, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)

    cv2.imwrite('./step3/mesona/left.png', np.concatenate((pic1, pic2), axis=1))

    pic1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt_x, pt_xp in zip(lp, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0 = 0
        y0 = (-r[2]/r[1]).astype(np.int)
        x1 = w
        y1 = (-(r[2]+r[0]*w)/r[0]).astype(np.int)
        pic2 = cv2.line(pic2, (x0,y0), (x1,y1), color, 1)
        pic1 = cv2.circle(pic1, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        pic2 = cv2.circle(pic2, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)

    cv2.imwrite('./step3/mesona/right.png', np.concatenate((pic1, pic2), axis=1))