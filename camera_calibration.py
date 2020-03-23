import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
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
### ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
### Vr = np.array(rvecs)
### Tr = np.array(tvecs)
### extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)


"""
Write your code here

## the goal is get the extrinsics and the mtx
## 1. get H
## 2. find B from H
## 3. B = K-1tK-1
"""
## H is Hi, the number of img is img_count
H = []
img_count = len(objpoints)

## step 1 : cal all the Homo matrix of each img, and retval is h
for iter_img in range(img_count):
    retval, mask = cv2.findHomography(cv2.UMat(objpoints[iter_img]), cv2.UMat(imgpoints[iter_img]))
    H.append(retval)

## step 2 : get B, V is the matrix in SVD
## U, Sigma, Vt are for U, sigma, Vt
V = np.zeros((2 * len(H), 6))

for iter_H in range(len(H)):
    Hi = H[iter_H]
    V[2 * iter_H] = np.array([Hi[0, 0] * Hi[0, 1],Hi[0, 0] * Hi[1, 1] + Hi[1, 0] * Hi[0, 1],Hi[1, 0] * Hi[1, 1],Hi[2, 0] * Hi[0, 1] + Hi[0, 0] * Hi[2, 1],Hi[2, 0] * Hi[1, 1] + Hi[1, 0] * Hi[2, 1],Hi[2, 0] * Hi[2, 1]])
    V[2 * iter_H + 1] = np.subtract(np.array([Hi[0, 0] * Hi[0, 0],Hi[0, 0] * Hi[1, 0] + Hi[1, 0] * Hi[0, 0],Hi[1, 0] * Hi[1, 0],Hi[2, 0] * Hi[0, 0] + Hi[0, 0] * Hi[2, 0],Hi[2, 0] * Hi[1, 0] + Hi[1, 0] * Hi[2, 0],Hi[2, 0] * Hi[2, 0]])
        , np.array([Hi[0, 1] * Hi[0, 1],Hi[0, 1] * Hi[1, 1] + Hi[1, 1] * Hi[0, 1],Hi[1, 1] * Hi[1, 1],Hi[2, 1] * Hi[0, 1] + Hi[0, 1] * Hi[2, 1],Hi[2, 1] * Hi[1, 1] + Hi[1, 1] * Hi[2, 1],Hi[2, 1] * Hi[2, 1]])
        )


U, Sigma, Vt = np.linalg.svd(V)
pre_B = Vt[np.argmin(Sigma)]
sym_B = np.zeros(9)
## sym_B is symmetic
sym_B[0] = pre_B[0]
sym_B[1] = sym_B[3] = pre_B[1]
sym_B[2] = sym_B[6] = pre_B[3]
sym_B[4] = pre_B[2]
sym_B[5] = sym_B[7] = pre_B[4]
sym_B[8] = pre_B[5]
sym_B = sym_B.reshape(3, 3)

# check eigenvalues are positive define or not
if not np.all(np.linalg.eigvals(sym_B) > 0):
    sym_B *= -1

## step 3 : sym_B = K-1tK-1
K_inverse = np.linalg.cholesky(sym_B)
K = np.linalg.inv(K_inverse.T)
## print origin K
print(K)
K[0][1] = 0.0
K[2][2] = 1.0

## get extrinsic [r1 r2 t]
extrinsic = []

for iter_img in range(img_count):
    tmp = np.matmul(K_inverse.T, H[iter_img][:, 0])
    lda = 1 / (np.sqrt(tmp.dot(tmp)))
    r1 = lda * tmp
    r2 = lda * np.matmul(K_inverse.T, H[iter_img][:, 1])
    r3 = np.cross(r1, r2)
    t = lda * np.matmul(K_inverse.T, H[iter_img][:, 2])
    extrinsic.append(np.column_stack((r1, r2, r3, t)))

mtx = K

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
