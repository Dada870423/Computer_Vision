import matplotlib.pyplot as plt
import numpy as np
import cv2

imname1 = 'data/1.jpg'
imname2 = 'data/2.jpg'

# part1 
img1 = cv2.imread(imname1)
Gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(imname2)
Gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(Gimg1, None)
kp2, des2 = sift.detectAndCompute(Gimg2, None)

# part2

#找match不確定可不可以用cv2 function
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.7*m[1].distance:
        good.append((m[0].trainIdx, m[0].queryIdx))
matches = np.asarray(good)

(hA, wA) = img1.shape[:2]
(hB, wB) = img2.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = img1
vis[0:hB, wA:] = img2

for (trainIdx, queryIdx) in matches:
    temp = np.random.randint(0, high=255, size=(3,))
    color = (np.asscalar(temp[0]), np.asscalar(temp[1]), np.asscalar(temp[2]))
    #print(color)
    ptA = (int(kp1[queryIdx].pt[0]), int(kp1[queryIdx].pt[1]))
    ptB = (int(kp2[trainIdx].pt[0] + wA), int(kp2[trainIdx].pt[1]))

    cv2.line(vis, ptA, ptB, color, 1)
plt.imshow(vis)
plt.show()

# part3 
#do RANSAC
n_times = 100
s = 4 #pick 4 points to find H
for it in range(n_times):
    sample_idx = np.random.choice(matches.size, s, replace=False)
    print(sample_idx)
    # find H
    H = np.zeros(9)
    
    
    
    