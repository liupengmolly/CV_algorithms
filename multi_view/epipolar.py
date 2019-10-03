import numpy as np

def lls_eight_points_alg(points1, points2):
    """
    利用最小二乘法结合奇异值分解对两张图像对应的8个甚至更多点计算基本矩阵。
    注意：F秩为2， 最后奇异值分解后需要选取前两个奇异值对应的重新计算。

    :param points1:(nx3)的齐次坐标
    :param points2:(nx3)的齐次坐标
    :return:
    """
    points_num = points1.shape[0]
    W = np.zeros((points_num,9))
    for i in range(points_num):
        u1 = points1[i][0]
        v1 = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]

        W[i] = np.array([u1*u2,u1*v2,u1,v1*u2,v1*v2,v1,u2,v2,1])

    U,s,VT = np.linalg.svd(W)
    f = VT[-1]
    F_hat = f.reshape(3,3)

    U_F,s_F_hat,VT_F = np.linalg.svd(F_hat)
    s_F = np.zeros((3,3))
    s_F[0,0],s_F[1,1] = s_F_hat[0], s_F_hat[1]
    F = U_F.dot(s_F).dot(VT_F)

    return F

def normalized_eight_points_alg(points1,points2):
    """
    利用最小二乘法结合奇异值分解对归一化的对应图像的八个甚至更多点的计算基本矩阵。
    :param points1: (nx3)的齐次坐标
    :param points2:(nx3)的齐次坐标
    :return:
    """

    N = points1.shape[0]

    points1_uv, points2_uv = points1[:,:2], points2[:,:2]
    mean1 = np.mean(points1_uv,axis=0)
    mean2 = np.mean(points2_uv,axis=0)

    points1_uv_center = points1_uv - mean1
    points2_uv_center = points2_uv - mean2

    scale1 = np.sqrt(np.sum(points1_uv_center**2)/N/2)
    scale2 = np.sqrt(np.sum(points2_uv_center**2)/N/2)

    T1 = np.array([[1/scale1,0,-mean1[0]/scale1],
                   [0,1/scale1,-mean1[1]/scale1],
                   [0, 0, 1]])

    T2 = np.array([[1/scale2,0,-mean2[0]/scale2],
                   [0,1/scale2,-mean2[1]/scale2],
                   [0, 0, 1]])
    points1_normalized = T1.dot(points1.T).T
    points2_normalized = T2.dot(points2.T).T

    Fq = lls_eight_points_alg(points1_normalized,points2_normalized)

    F = T1.T.dot(Fq).dot(T2)

    return F


def compute_distance_to_epipolar_lines(points1, points2, F):
    l = F.dot(points2.T)
    dis_sum = 0.0

    for i in range(8):
        dis =  np.abs(points1[i][0]*l[0][i]+points1[i][1]*l[1][i]+l[2][i])*1.0/np.sqrt(l[0][i]**2+l[1][i]**2)
        dis_sum += dis

    return dis_sum/8


def compute_epipole(points1, points2, F):
    l = F.dot(points2.T).T
    U,s,VT = np.linalg.svd(l)
    e = VT[-1]
    e/=e[2]
    return e

def compute_matching_homographies(e2,F,im2,points1,points2):
    """
    计算能矫正一对图片的单应H1，H2, H2通过构建特定的使e2为无穷远点的矩阵得到，H1通过与H2使用
    最小二乘法得到

    :param e2:
    :param F:
    :param im2:
    :param points1:
    :param points2:
    :return:
    """
    width = im2.shape[1]
    height = im2.shape[0]

    T = np.identity(3)
    T[0,2] = -width/2
    T[1,2] = -height/2

    e = T.dot(e2)
    e1_prime = e[0]
    e2_prime = e[1]
    e3_prime = e[2]
    alpha = 1.0 if e1_prime >= 0 else -1.0
    R = np.identity(3)
    R[0,0] = alpha*e1_prime/np.sqrt(e1_prime**2+e2_prime**2)
    R[0,1] = alpha*e2_prime/np.sqrt(e1_prime**2+e2_prime**2)
    R[1,0] = -alpha*e2_prime/np.sqrt(e1_prime**2+e2_prime**2)
    R[1,1] = alpha*e1_prime/np.sqrt(e1_prime**2+e2_prime**2)

    f = R.dot(e)[0]
    G = np.identity(3)
    G[2,0] = -1/f

    H2 = np.linalg.inv(T).dot(G.dot(R).dot(T))

    e_prime = np.array([[0,-e2[2],e2[1]],
                        [e2[2],0,-e2[0]],
                        [-e2[1],e2[0],0]])
    v = np.ones((1,3))
    M = e_prime.dot(F.T) + np.outer(e2,v)
    # M = e_prime.dot(F)

    points1_hat = H2.dot(M.dot(points1.T)).T
    points2_hat = H2.dot(points2.T).T

    W = points1_hat/points1_hat[:,2].reshape(-1,1)
    b = (points2_hat/points2_hat[:,2].reshape(-1,1))[:,0]
    a1,a2,a3 = np.linalg.lstsq(W,b)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1,a2,a3])

    H1 = HA.dot(H2).dot(M)

    return H1,H2
