import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

############################# Magic.......
def Magic(k, l, h):
    MagicV = np.array([h[0, k]*h[0, l],
                     h[0, k]*h[1, l] + h[1, k]*h[0, l],
                     h[1, k]*h[1, l],
                     h[2, k]*h[0, l] + h[0, k]*h[2, l],
                     h[2, k]*h[1, l] + h[1, k]*h[2, l],
                     h[2, k]*h[2, l]], 
                     dtype='float32')
    return MagicV




#############################




# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""


"""
Write your code here
## 1. get H
## 2. get B 
## 3. B=KtK
## 4. ex = H K 


"""

## get H
###
H = np.zeros((len(images), 9))

for iter_H in range(len(objpoints)):
    Obj_Point = objpoints[iter_H]
    Img_point = imgpoints[iter_H]
    M = np.zeros((2 * len(Obj_Point), 9), dtype = np.float64)
    
    Img_point = Img_point.reshape(49, 2)
    Img_point = np.concatenate((Img_point, np.ones(shape = (49, 1))), axis = 1)
    
    for iter_M in range(len(Obj_Point)):
        x = Obj_Point[iter_M, 0]
        y = Obj_Point[iter_M, 1]
        u = Img_point[iter_M, 0]
        v = Img_point[iter_M, 1]
        
        M[iter_M * 2] = np.array([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        M[iter_M * 2 + 1] = np.array([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    U_H, Sigma_H, Vt_H = np.linalg.svd(M)
    index_Min = np.argmin(Sigma_H)
    
    ##Norm it
    for iter_Vt in range(9):
        Vt_H[index_Min, iter_Vt] = Vt_H[index_Min, iter_Vt] / Vt_H[index_Min, 8]
        
    H[iter_H] = Vt_H[index_Min]
    
    
H = np.reshape(H, (len(images), 3, 3))
print("H: ", H)
### H done

## get B
MagicV = np.zeros([2 * len(H), 6])
index = 0
for iter_H in H:
    MagicV[index] = Magic(0, 1, iter_H)
    MagicV[index + 1] = Magic(0, 0, iter_H) - Magic(1, 1, iter_H)
    index = index + 2

U_B, Sigma_B, VT_B = np.linalg.svd(MagicV)
index_Min_B = np.argmin(Sigma_B)
b = VT_B[index_Min_B]
print("b; ", b)
B = np.array([[b[0],b[1],b[3]],
              [b[1],b[2],b[4]],
              [b[3],b[4],b[5]]])

Oy = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])

if b[0] == 0:
    b[0] = 1.0
Lda_B = b[5] - (b[3] * b[3] + Oy * (b[1] * b[3] - b[0] * b[4])) / b[0]
B = B / Lda_B

## B done 
## Calculate K 
K = np.linalg.inv(np.linalg.cholesky(B).transpose())
print("K:")
print(K)
## K done

## do extrinsics

extrinsics = np.zeros((len(H), 3, 4))
K_inverse = np.linalg.inv(K)

for index in range(len(images)):
    h = H[index]  
    h1 = h[:, 0]
    h2 = h[:, 1]
    h3 = h[:, 2]
    
    Lda_1 = 1 / np.linalg.norm(np.dot(K_inverse,h1), ord = 2)
    Lda_2 = 1 / np.linalg.norm(np.dot(K_inverse,h2), ord = 2)
    Lda_3 = (Lda_1 + Lda_2) / 2
    
    r1 = Lda_1 * (np.dot(K_inverse,h1))
    r2 = Lda_2 * (np.dot(K_inverse,h2))
    r3 = np.cross(r1, r2)
    t = Lda_3 * (np.dot(K_inverse,h3))
    extrinsics[index] = np.array([[r1[0],r2[0],r3[0],t[0]],
                                  [r1[1],r2[1],r3[1],t[1]],
                                  [r1[2],r2[2],r3[2],t[2]]])

## extrinsics done


## set mtk, extrinsisc

mtx = K
# extrinsics define before

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
