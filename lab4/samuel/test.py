import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import inv
import random
def l2_distance(a,b):
    return np.linalg.norm(a-b)

def detectAndDescribe(image):
        # Change colorful to gray
        #gray = (0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]).astype(np.uint8) # Y = 0.299R + 0.587G + 0.114B
        
        # SIFT create
        descriptor = cv2.xfeatures2d.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)

        return keypoints, features

def bfmatch(input1, input2):
    #print(input1.shape)
    #print(input2.shape)
    res = []
    for ii in range(input1.shape[0]):
        min_dist = float('inf')
        min_trainIdx = 0
        min_queryIdx = 0
        for jj in range(input2.shape[0]):
            tmp_dist = l2_distance(input1[ii], input2[jj])
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_trainIdx = ii
                min_queryIdx = jj
        tmp = cv2.DMatch(_distance=min_dist, _trainIdx=min_queryIdx, _queryIdx=min_trainIdx, _imgIdx=0)
        res.append(tmp)
    return res


def matchKeypoints(keypointsA, keypointsB, featuresA, featuresB, ratio, reprojThresh, imageA, imageB):
	matches = bfmatch(featuresA, featuresB)
	matches = sorted(matches, key=lambda x: x.distance)
	keypoints = [keypointsA, keypointsB]

	"""
	    match_pic=0
		match_pic=cv2.drawMatches(imageA,keypointsA,imageB,keypointsB,matches[:30], match_pic,flags=2)
		cv2.imshow("match_pic",match_pic)
		cv2.waitKey(0)
	"""
	return matches


def normalize(keypoints_1,keypoints_2,matches):
	points_array1=[]
	points_array2=[]
	#print (type(matches))
	for i in range(0,8):
		x1,y1 = keypoints_1[matches[i].queryIdx].pt
		x2,y2 = keypoints_2[matches[i].trainIdx].pt
		arr1=np.array([x1,y1,1])
		arr2=np.array([x2,y2,1])
		points_array1.append(arr1)
		points_array2.append(arr2)

	points_array1 = np.asarray(points_array1)
	points_array2 = np.asarray(points_array2)

	#print(points_array1.shape)
	#print(points_array1)

	
	S1 = np.sqrt(2) / np.std(points_array1)
	mean_1 = np.mean(points_array1,axis=0)
	#print("mean 1 is %%",mean_1)
	T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])

	S2 = np.sqrt(2) / np.std(points_array2)
	mean_1 = np.mean(points_array2,axis=0)
	T2 = np.array([[S2,0,-S2*mean_1[0]],[0,S2,-S2*mean_1[1]],[0,0,1]])

	for i in range(0,8):
		points_array1[i] = np.dot(T1,points_array1[i])
		points_array2[i] = np.dot(T2,points_array2[i])


	#print(points_array1)
	#print(points_array2)
    	
	return points_array1,points_array2

def second_normalized(keypoints_1,keypoints_2,matches,img1,img2):
	shape_x_1 = img1.shape[0]
	shape_x_2 = img2.shape[0]
	shape_y_1 = img1.shape[1]
	shape_y_2 = img2.shape[1]
	#shape_x_1,shape_y_1,dumb=img1.shape
	#shape_x_2,shape_y_2,dumb=img2.shape

	points_array1=[]
	points_array2=[]

	

	T_1 = np.array([[(2.0/shape_x_1),0,-1],[0,(2.0/shape_y_1),-1],[0,0,1]],dtype=float)	#transformation array
	T_2 = np.array([[(2.0/shape_x_2),0,-1],[0,(2.0/shape_y_2),-1],[0,0,1]],dtype=float)

	#print(T_1)
	#print(T_2)

	mid_x_1,mid_y_1 = shape_x_1/2,shape_y_1/2
	mid_x_2,mid_y_2 = shape_x_2/2,shape_y_2/2

	for i in range(0,8):
		x1,y1 = keypoints_1[matches[i].queryIdx].pt
		x2,y2 = keypoints_2[matches[i].trainIdx].pt
		#x1,y1 = (x1-mid_x_1)/shape_x_1, (y1-mid_y_1)/shape_y_1
		#x2,y2 = (x2-mid_x_2)/shape_x_2, (y2-mid_y_2)/shape_y_2
		#arr1=np.array([x1,y1,1])
		arr1=np.array(np.dot(T_1,[x1,y1,1]))
		arr2=np.array(np.dot(T_2,[x2,y2,1]))
		#arr2=np.array([x2,y2,1])
		points_array1.append(arr1)
		points_array2.append(arr2)

	points_array1 = np.asarray(points_array1)
	points_array2 = np.asarray(points_array2)

	#print("==========")
	#print(points_array1)
	#print(points_array2)
	#print("==========")
	return points_array1,points_array2,T_1,T_2

def denormalize(T1,T2,fundamental):
	return np.dot(np.dot(T2.T,fundamental),T1)

def elementary_matrix(keypoints_1,keypoints_2,matches,img1,img2):
	#print("enter elementary_matrix calculation")
	pts1,pts2,T1,T2=second_normalized(keypoints_1,keypoints_2,matches,img1,img2)

	n=8
	A = np.zeros((n,9))
	B = np.zeros((n,9))

	for i in range(0,n):
		A[i] = [pts1[i][0]*pts2[i][0], pts1[i][0]*pts2[i][1], pts1[i][0]*pts2[i][2],
				pts1[i][1]*pts2[i][0], pts1[i][1]*pts2[i][1], pts1[i][1]*pts2[i][2],
				pts1[i][2]*pts2[i][0], pts1[i][2]*pts2[i][1], pts1[i][2]*pts2[i][2]]
		B[i] = [pts1[i][0]*pts2[i][0], pts1[i][0]*pts2[i][1], pts1[i][0],
				pts1[i][1]*pts2[i][0], pts1[i][1]*pts2[i][1], pts1[i][1],
				pts2[i][0], pts2[i][1],1]
            
	#compute linear least square solution
	U,S,V = np.linalg.svd(A)
	F = V[-1].reshape(3,3)	#Ax=0
	
	U,S,V = np.linalg.svd(B)
	ans = V[-1].reshape(3,3)

    # constrain F
    # make rank 2 by zeroing out last singular value
	U,S,V = np.linalg.svd(F)
	S[2] = 0
	F = np.dot(U,np.dot(np.diag(S),V))

	U,S,V = np.linalg.svd(ans)
	S[2] = 0
	ans = np.dot(U,np.dot(np.diag(S),V))

	#print("before denormalize")
	#print(ans)
	##print(np.linalg.det(ans))
	fundamental = denormalize(T1,T2,ans)

	##print(ans/ans[2,2])
	#print("after denormalize")
	#print(fundamental)
	##print(np.linalg.det(fundamental))

	x2,y2=keypoints_1[matches[1].queryIdx].pt
	temp=np.array([x2,y2,1])
	##print("the parameter of epiline is : ",np.dot(fundamental,temp))
	epiline = np.dot(fundamental.T,temp)
	#print(temp,epiline)
	x = (-1*epiline[2])/epiline[0]	#x,0
	y = (-1*epiline[2])/epiline[1]	#100,y
	#print("x:{},y:{}",x,y)



	return fundamental

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r = img1.shape[0]
    c = img1.shape[1]
    #r,c = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2



img1 = cv2.imread('Mesona1.JPG')  #queryimage # left image
img2 = cv2.imread('Mesona2.JPG') #trainimage # right image
'''
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
'''
#print("Doing SIFT on imgA")
(keypointsA, featuresA) = detectAndDescribe(img1)
#print("Doing SIFT on imgB")
(keypointsB, featuresB) = detectAndDescribe(img2)
#print("Done")


match = matchKeypoints(keypointsA, keypointsB, featuresA, featuresB, 0.75, 4.0, img1, img2)



sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print(F)
print("===")


fundamental = elementary_matrix(keypointsA,keypointsB,match,img1,img2)
print(fundamental)
# We select only inlier points
#pts1 = pts1[mask.ravel()==1]
#pts2 = pts2[mask.ravel()==1]
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()