import numpy as np
import cv2
import argparse # You can test our different datas on cmd, friendly to users
import matplotlib.pyplot as plt  # For visualization
import glob # For FILE I/O reading chessboards(As well as hw1 provided by TAs)
import matplotlib.tri as mtri # Used in obj_main.m in python version
from mpl_toolkits.mplot3d import Axes3D # For visualization
def show_image(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.xticks([]), plt.yticks([])

    plt.show()

def drawlines(img1, img2, lines, pts1, pts2):
    print(img1.shape)
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    t_pts1 = np.int32(pts1)
    t_pts2 = np.int32(pts2)
    for r, pt1, pt2 in zip(lines, t_pts1, t_pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1,lineType=4)
        img1 = cv2.circle(img1,tuple(pt1[0:2]),5,color,-1)
        img2 = cv2.circle(img2, tuple(pt2[0:2]), 5, color,-1)
    return img1, img2

def l2_distance(a,b):
    return np.linalg.norm(a-b)


def detectAndDescribe(image):
    # SIFT create
    descriptor = cv2.xfeatures2d.SIFT_create()
    keypoints, features = descriptor.detectAndCompute(image, None)

    return keypoints, features


def bfmatch(input1, input2):
    print(input1.shape)
    print(input2.shape)
    res = []
    for ii in range(0, input1.shape[0]):
        min_dist = float('inf')
        min_trainIdx = 0
        min_queryIdx = 0
        for jj in range(input2.shape[0]):
            tmp_dist = l2_distance(input1[ii], input2[jj])
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_trainIdx = ii
                min_queryIdx = jj
        # Using struct, allowed by TA ( tiny.cc/kim24y )
        tmp = cv2.DMatch(_distance=min_dist, _trainIdx=min_queryIdx, _queryIdx=min_trainIdx, _imgIdx=0) 
        res.append(tmp)
    return res

def normalize2dpts(in_pts):
    centroid = np.array([in_pts[:, 0].mean(), in_pts[:, 1].mean(), 0])
    pts = in_pts - centroid

    meandist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).mean()
    scale = np.sqrt(2) / (meandist)

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    newpts = np.dot(T, in_pts.transpose()).transpose()
    #     result = np.dot(np.linalg.inv(T), newpts.transpose()).transpose()
    return newpts, T


def fundmat(in_points_in_img1, in_points_in_img2):
    # normalize
    points_in_img1, T1 = normalize2dpts(in_points_in_img1)
    points_in_img2, T2 = normalize2dpts(in_points_in_img2)

    # Solve for A
    s = points_in_img1.shape[0]

    A = np.zeros((s, 9))
    for index in range(0, s):
        x, y = points_in_img1[index][0], points_in_img1[index][1]
        tx, ty = points_in_img2[index][0], points_in_img2[index][1]
        A[index] = [tx * x, tx * y, tx, ty * x, ty * y, ty, x, y, 1]

    u, s, v = np.linalg.svd(A)
    F = v[-1].reshape(3, 3)  # eigenvector with the least eigenvalue

    u, s, v = np.linalg.svd(F)
    s[2] = 0
    F = np.dot(np.dot(u, np.diag(s)), v)

    # denormalize
    F = np.dot(np.dot(T2.transpose(), F), T1)

    return F / F[2, 2]


def computeEpipoleLines(F, pts):
    lines = np.dot(F, pts.transpose()).transpose()
    n = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).reshape(-1, 1)
    return lines / n * -1


def transformPtsToArrayPts(kp1, kp2, matches):
    tup_matches_kp1 = [kp1[dt.queryIdx].pt for dt in matches]
    tup_matches_kp2 = [kp2[dt.trainIdx].pt for dt in matches]
    matches_kp1 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp1])
    matches_kp2 = np.array([[h for h in kp] + [1] for kp in tup_matches_kp2])
    return matches_kp1, matches_kp2


def calculateSampsonDistance(matches_kp1, matches_kp2, F):
    Fx1 = np.dot(F, matches_kp1.transpose())
    Fx2 = np.dot(F.transpose(), matches_kp2.transpose())
    denom = (Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2).reshape(-1, 1)
    err = (np.diag(np.dot(matches_kp2, np.dot(F, matches_kp1.transpose()))) ** 2)
    err = err.reshape(-1, 1) / denom
    return err


def randomPartition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def findFundMatRansac(matches_kp1, matches_kp2, s=8, threshold=1,
                      maxIterations=10000, returnMatches=False,
                      inlierThreshold=50, confidence=0.99999):
    cnt_matches = matches_kp1.shape[0]
    best_fit = []
    best_error = np.Infinity
    best_kp1, best_kp2 = [], []
    best_total = 0

    k = maxIterations
    for iter in range(k):
        maybe_idxs, test_idxs = randomPartition(s, cnt_matches)
        # Take s data points
        data_p1 = np.take(matches_kp1, maybe_idxs, axis=0)
        data_p2 = np.take(matches_kp2, maybe_idxs, axis=0)
        # Fit a fundamental matrix
        F = fundmat(data_p1, data_p2)

        # Test the current fundamental matrix
        test_p1 = np.take(matches_kp1, test_idxs, axis=0)
        test_p2 = np.take(matches_kp2, test_idxs, axis=0)
        errs = calculateSampsonDistance(test_p1, test_p2, F)

        # Current Inliers
        inlier_indices = [errs[:, 0] < threshold]

        # Get Current Inliers
        current_p1 = np.append(data_p1, test_p1[tuple(inlier_indices)], axis=0)
        current_p2 = np.append(data_p2, test_p2[tuple(inlier_indices)], axis=0)
        current_total = current_p1.shape[0]

        if current_total > best_total and current_total >= inlierThreshold:
            better_fit = fundmat(current_p1, current_p2)
            better_err = calculateSampsonDistance(current_p1, current_p2, F)

            if (best_error > better_err.mean()):
                best_fit = better_fit
                best_kp1 = current_p1
                best_kp2 = current_p2
                best_total = current_p1.shape[0]

                # # we are done in case we have enough inliers
                r = current_total / cnt_matches
                nk = np.log(1 - confidence) / np.log(1 - pow(r, s))
                k = iter + nk

        if iter > k:
            break
    #     print(str(best_total) + "/" + str(cnt_matches))
    if returnMatches:
        return best_fit, best_kp1, best_kp2

    return best_fit


def decomposeEssentialMatrix(E):
    u, s, v = np.linalg.svd(E)
    m = (s[0] + s[1]) / 2
    E = np.dot(np.dot(u, np.diag([m, m, 0])), v)

    u, s, v = np.linalg.svd(E)
    w = np.array([[0, -1, -0], [1, 0, 0], [0, 0, 1]])

    if np.linalg.det(v) < 0:
        v *= -1
    if np.linalg.det(u) < 0:
        u *= -1

    u3 = u[:, -1]
    R1 = np.dot(u, np.dot(w, v))
    R2 = np.dot(u, np.dot(w.transpose(), v))

    return [np.vstack((R1.transpose(), u3)).transpose(),
            np.vstack((R1.transpose(), -u3)).transpose(),
            np.vstack((R2.transpose(), u3)).transpose(),
            np.vstack((R2.transpose(), -u3)).transpose()]


def triangulatePoint(point_1, point_2, p1, p2):
    # define A
    u1, v1 = point_1[0], point_1[1]
    u2, v2 = point_2[0], point_2[1]

    A = np.zeros((4, 4))

    A[0] = u1 * p1[2] - p1[0]
    A[1] = v1 * p1[2] - p1[1]
    A[2] = u2 * p2[2] - p2[0]
    A[3] = v2 * p2[2] - p2[1]

    u, s, v = np.linalg.svd(A)
    x = v[-1]
    x = x / x[-1]
    return x


def triangulate(pts1, pts2, p1, p2):
    R1, t1 = getRmatAndTmat(p1)
    R2, t2 = getRmatAndTmat(p2)

    # compute camera centers
    C1 = -np.dot(R1.transpose(), t1)
    C2 = -np.dot(R2.transpose(), t2)

    V1 = np.dot(R1.transpose(), np.array([0, 0, 1]))
    V2 = np.dot(R2.transpose(), np.array([0, 0, 1]))

    points = []
    for pt1, pt2 in zip(pts1, pts2):
        point_in_3d = triangulatePoint(pt1, pt2, p1, p2)[:3]
        test1 = np.dot((point_in_3d - C1), V1)
        test2 = np.dot((point_in_3d - C2), V2)
        if (test1 > 0 and test2 > 0):
            points.append(point_in_3d)

    return np.array(points)


def getRmatAndTmat(p):
    R = p[:, :3]
    t = p[:, 3]
    return R, t


def findPandX(kp1, kp2, K1, K2, p2_solutions):
    p1 = np.vstack((np.eye(3), np.zeros(3))).transpose()
    p1 = np.dot(K1, p1)
    best_p2 = -1
    best_p2_inliers = -1
    best_p2_points = []
    for sol_p2 in p2_solutions:
        p2 = np.dot(K2, sol_p2)
        points = triangulate(kp1, kp2, p1, p2)
        if (best_p2_inliers < points.shape[0]):
            best_p2_inliers = points.shape[0]
            best_p2 = p2
            best_p2_points = points
    return p1, best_p2, best_p2_points

def CheckVisible(M, P1, P2, P3):
    '''
    used to check if the surface normal facing the camera

    M: 3x4 projection matrix
    P1, P2, P3: 3D points
    '''
    tri_normal = np.cross((P2 - P1), (P3 - P2))
    # camera direction
    cam_dir = np.asarray([M[2, 0], M[2, 1], M[2, 2]])

    test_result = np.dot(cam_dir, tri_normal)

    if (test_result < 0):
        bVisible = 1  # visible
    else:
        bVisible = 0  # invisible

    return bVisible


def obj_main(P, p_img2, M, tex_name, im_index):
    # Due to the Matlab API is trou

    tuples_img2_pts = [(p_img2[i, 0], p_img2[i, 1]) for i in range(len(p_img2))]
    img = plt.imread(tex_name)
    img_size = img.shape
    '''
    % mesh-triangulation
    '''
    tri = mtri.Triangulation(p_img2[:, 0], p_img2[:, 1]) # The same in obj_main.m -> delaunay, trisurf
    # trisurf mesh triangulation
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tri.triangles)

    with open('model' + str(im_index) + '.obj', 'w') as fp:
        fp.write("# objfile\n")
        fp.write('mtllib model' + str(im_index) + '.mtl\n\n')
        fp.write('usemtl Texture\n')

        for i in range(len(P)):
            fp.write('v %f %f %f \n' % (P[i, 0], P[i, 1], P[i, 2]))

        fp.write('\n\n\n')

        for i in range(len(p_img2)):
            fp.write('vt %f %f\n' % (p_img2[i, 0] / img_size[1], 1 - p_img2[i, 1] / img_size[0]))

        fp.write('\n\n\n')

        for i, triangle in enumerate(tri.triangles):
            #triangle[0] = min(triangle[0], P.shape[0]-1)
            #triangle[1] = min(triangle[1], P.shape[0]-1)
            #triangle[2] = min(triangle[2], P.shape[0]-1)
            bVisible = CheckVisible(M, P[triangle[0], :], P[triangle[1], :], P[triangle[2], :])
            if bVisible == True:
                fp.write('f %d/%d %d/%d %d/%d\n' % (
                triangle[0] + 1, triangle[0] + 1, triangle[1] + 1, triangle[1] + 1, triangle[2] + 1, triangle[2] + 1))
            else:
                fp.write('f %d/%d %d/%d %d/%d\n' % (
                triangle[1] + 1, triangle[1] + 1, triangle[0] + 1, triangle[0] + 1, triangle[2] + 1, triangle[2] + 1))

    with open('model' + str(im_index) + '.mtl', 'w') as fp:
        fp.write('# MTL file\n')
        fp.write('newmtl Texture\n')
        fp.write('Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n')
        fp.write('map_Kd ' + tex_name + '\n')

def GetSelfCameraMatrix():
    # The code in hw1
    corner_x = 7
    corner_y = 7
    objp = np.zeros((corner_x*corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    # To get the camera matrix of our camera and test on our data
    images = glob.glob('data/chessboard/*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
            #plt.imshow(img)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=int, default=1)
    args = parser.parse_args()

    # Test Data 1 (Mesona) 助教提供的測資1
    if args.dataset == 1:
        img1 = cv2.imread("Mesona1.JPG", 0)
        img2 = cv2.imread("Mesona2.JPG", 0)
        K1 = np.array([[1.4219, 0.0005, 0.5092], [0, 1.4219, 0.3802], [0, 0, 0.0010]])
        K2 = np.array([[1.4219, 0.0005, 0.5092], [0, 1.4219, 0.3802], [0, 0, 0.0010]])

    # Test Case 2 (Statue) 助教提供的測資2
    elif args.dataset == 2:
        img1 = cv2.imread("Statue1.bmp", 0)
        img2 = cv2.imread("Statue2.bmp", 0)
        K1 = np.array([[5426.566895, 0.678017, 330.096680], [0.000000, 5423.133301, 648.950012], [0, 0, 1]])
        K2 = np.array([[5426.566895, 0.678017, 387.430023], [0.000000, 5423.133301, 620.616699], [0, 0, 1]])

    # Self Data 自己的測資
    elif args.dataset == 3:
        img1 = cv2.imread("box1.jpg", 0)
        img2 = cv2.imread("box2.jpg", 0)
        #K1 = GetSelfCameraMatrix()
        K1 = np.array([[1.13290177e+03,0.00000000e+00, 3.46328702e+02],
                       [0.00000000e+00, 1.12878020e+03, 7.31892626e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        K2 = np.array([[1.13290177e+03, 0.00000000e+00, 3.46328702e+02],
                       [0.00000000e+00, 1.12878020e+03, 7.31892626e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # Self Data 自己的測資
    elif args.dataset == 4:
        img1 = cv2.imread("toothpaste1.jpg", 0)
        img2 = cv2.imread("toothpaste2.jpg", 0)
        #K1 = GetSelfCameraMatrix()
        K1 = np.array([[1.13290177e+03,0.00000000e+00, 3.46328702e+02],
                       [0.00000000e+00, 1.12878020e+03, 7.31892626e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        K2 = np.array([[1.13290177e+03, 0.00000000e+00, 3.46328702e+02],
                       [0.00000000e+00, 1.12878020e+03, 7.31892626e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    else:
        print('Wrong argument')
        exit(1)


    keypointsA, featuresA = detectAndDescribe(img1)
    keypointsB, featuresB = detectAndDescribe(img2)
    print(" SIFT Done")

    matches = bfmatch(featuresA, featuresB)

    # Find The Fundamental Matrix

    matches_keypointsA, matches_keypointsB = transformPtsToArrayPts(keypointsA, keypointsB, matches)

    F, best_keypointsA, best_keypointsB = findFundMatRansac(matches_keypointsA, matches_keypointsB, returnMatches=True, threshold=0.1)

    # Show Our Correspondences
    lines1 = computeEpipoleLines(F.transpose(), best_keypointsB)
    img3, img4 = drawlines(img1, img2, lines1, best_keypointsA, best_keypointsB)
    show_image(img3, 'Epipolar Lines')
    show_image(img4, 'Interest Points')
    lines2 = computeEpipoleLines(F, best_keypointsA)
    img6, img5 = drawlines(img2, img1, lines2, best_keypointsB, best_keypointsA)

    # Computing Essential Matrix

    E = np.dot(K2.transpose(), np.dot(F, K1))

    # Find The Best Camera Matrix P and 3d-Points

    all_p2_solutions = decomposeEssentialMatrix(E) # Four possible P2
    best_p1, best_p2, points = findPandX(best_keypointsA, best_keypointsB, K1, K2, all_p2_solutions)
    if args.dataset == 1:
        obj_main(points, best_keypointsA, best_p1, "Mesona1.JPG", 1)
    elif args.dataset == 2:
        obj_main(points, best_keypointsA, best_p1, "Statue1.bmp", 2)
    elif args.dataset == 3:
        obj_main(points, best_keypointsA, best_p1, "box1.jpg", 3)
    elif args.dataset == 4:
        obj_main(points, best_keypointsA, best_p1, "toothpaste1.jpg", 4)
    plt.rcParams['figure.figsize'] = [8, 8]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    plt.show()