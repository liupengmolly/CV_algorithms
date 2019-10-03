import numpy as np
from multi_view.triangulation import nonlinear_estimate_3d_point

def factorization_method(points_im1, points_im2):
    """
    The Tomasi and Kanade Factorization Method to determine the 3D structure of the scene
    and the motion of the cameras.
    :param points_im1:
    :param points_im2:
    :return:
    """
    N = points_im1.shape[0]
    points_set = [points_im1, points_im2]
    D = np.zeros((4,N))

    for i in range(len(points_set)):
        points = points_set[i]
        centroid = 1.0/N*points.sum(axis=0)
        points[:,0] -= centroid[0]*np.ones(N)
        points[:,1] -= centroid[1]*np.ones(N)

        D[2*i:2*i+2,:] = points[:,0:2].T

    U,s,VT = np.linalg.svd(D)
    M = U[:,:3]
    S = np.diag(s)[0:3,0:3].dot(VT[0:3,:])
    return M,S

def estimate_initial_RT(E):
    """
    通过本质矩阵求摄像机外参，该函数的使用一般在已知摄像机内参的情况下，只需要根据一对图像中的若干
    对应点计算出本质矩阵后，计算摄像机外参后就能得到摄像机矩阵
    :param E:
    :return:
    """
    U,s,VT = np.linalg.svd(E)

    Z = np.array([[0, 1, 0],
                  [-1,0, 0],
                  [0, 0, 0]])
    W = np.array([[0,-1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R = U.dot(W).dot(VT)
    R_reverse = U.dot(W.T).dot(VT)

    R = np.linalg.det(R)*R
    R_reverse = np.linalg.det(R_reverse)*R_reverse

    T1 = U[:,2].reshape(-1,1)
    T2 = -U[:,2].reshape(-1,1)

    R_set = [R, R_reverse]
    T_set = [T1, T2]

    RT = np.zeros((4,3,4))
    for i in range(len(R_set)):
        for j in range(len(T_set)):
            RT[i*2+j] = np.hstack((R_set[i],T_set[j]))

    return RT

def camera1_to_camera2(P, RT):
    """
    将摄像机1坐标系下的3D坐标P转换为摄像机2坐标系下的3D坐标。
    :param P:
    :param RT:
    :return:
    """
    P_homo = np.array([P[0], P[1], P[2], 1.0])
    A = np.zeros((4,4))
    A[0:3,:] = RT
    A[3,:] = np.array([0.0,0.0,0.0,1.0])
    P_prime_homo = A.dot(P_homo.T)
    P_prime_homo /= P_prime_homo[3]
    P_prime = P_prime_homo[0:3]
    return P_prime

def estimate_RT_from_E(E, image_points, K):
    """
    对estimate_initial_RT得到的4对可能的RT，选择在摄像机正面（由两个摄像机矩阵计算得到的3D空间中的
    Z坐标都为正）最多的点对应的RT

    :param E:
    :param image_points:N measured points in each of the M images(NxMx2)
    :param K:
    :return:
    """
    RT = estimate_initial_RT(E)
    count = np.zeros((1,4))
    IO = np.hstack((np.identity(3),np.zeros((3,1))))
    M1 = K.dot(IO)

    camera_matrixes = np.zeros((2,3,4))
    camera_matrixes[0] = M1
    for i in range(RT.shape[0]):
        RTi = RT[i]
        M2i = K.dot(RTi)
        camera_matrixes[1] = M2i
        for j in range(image_points.shape[0]):
            pointj_3d = nonlinear_estimate_3d_point(image_points[j], camera_matrixes)
            Pj = np.vstack((pointj_3d.reshape(3,1),[1]))
            Pj_prime = camera1_to_camera2(Pj,RTi)
            if Pj[2]>0 and Pj_prime[2]>0:
                count[0,i] += 1

    maxIndex = np.argmax(count)
    maxRT = RT[maxIndex]

    return maxRT



