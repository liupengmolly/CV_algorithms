import numpy as np
from utils.coordinate import reprojection_error


def linear_estimate_3d_point(image_points, camera_matrixes):
    """
    在已知M个摄像机矩阵和对应的成像点的坐标后，通过线性方法求3D空间中原始点(只一个)的非齐次坐标。

    :param image_points:3D空间的点在每个摄像机成像平面中的坐标。Mx2
    :param camera_matrixes: M个相机的摄像机矩阵，Mx3x4
    :return:
    """
    N = image_points.shape[0]
    A = np.zeros((2*N, 4))

    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrixes[i]
        Aix = pi[0]*Mi[2] - Mi[0]
        Aiy = pi[1]*Mi[2] - Mi[1]
        A[i*2] = Aix
        A[i*2+1] = Aiy

    U,s,VT = np.linalg.svd(A)
    P_homo = VT[-1]
    P_homo /= P_homo[-1]
    P = P_homo[:3]

    return P

def jacobian(point_3d, camera_matrices):
    J = np.zeros((2*camera_matrices.shape[0], 3))
    point_3d_homo = np.hstack((point_3d, 1))
    J_set = []

    for i in range(camera_matrices.shape[0]):
        Mi = camera_matrices[i]
        pi = Mi.dot(point_3d_homo)
        Jix = (pi[2]*np.array([Mi[0, 0], Mi[0, 1], Mi[0, 2]]) \
               - pi[0]*np.array([Mi[2, 0], Mi[2, 1], Mi[2, 2]])) / pi[2]**2
        Jiy = (pi[2]*np.array([Mi[1, 0], Mi[1, 1], Mi[1, 2]]) \
               - pi[1]*np.array([Mi[2, 0], Mi[2, 1], Mi[2, 2]])) / pi[2]**2
        J_set.append(Jix)
        J_set.append(Jiy)

    for i in range(J.shape[0]):
        J[i] = J_set[i]
    return J

def nonlinear_estimate_3d_point(image_points, camera_matrixes, iterate_num=10):
    """
    通过非线性优化(用牛顿迭代法）求给定摄像机矩阵和对应成像平面上点的坐标，求3D空间中点的坐标

    :param image_points:
    :param camera_matrixes:
    :param iterate_num: 默认为10
    :return:
    """

    P = linear_estimate_3d_point(image_points, camera_matrixes)

    for i in range(iterate_num):
        e = reprojection_error(P, image_points,camera_matrixes)
        J = jacobian(P, camera_matrixes)
        P -= np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)

    return P
