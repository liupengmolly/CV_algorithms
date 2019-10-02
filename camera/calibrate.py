import numpy as np


def calibrate_affine_camera_matrix(XY_3D1,XY_3D2,img1,img2):
    """
    通过特定用来标定的物体在真实3D中的位置以及成像后的位置，来标定摄像机的仿射矩阵。
    注意，由于用来标定相机的物体为一个平面，缺少深度信息，所以需要该物体在不同深度
    位置的成像来标定最终的仿射矩阵

    :param XY_3D1:真实3D空间中深度为Z1的齐次坐标,(img_num1,4)
    :param XY_3D2:真实3D空间中深度为Z2的齐次坐标,(img_num2,4)
    :param img1: 物体在深度为Z1时的成像2D齐次坐标，(img_num1,3)
    :param img2: 物体在深度为Z2时的成像2D齐次坐标，(img_num2,3)

    :return:摄像机仿射矩阵，narray(3x4), 最后一行为（0，0，0，1）
    """
    img_num1 = img1.shape[0]
    img_num2 = img2.shape[0]

    A = np.zeros((2*(img_num1+img_num2),8))
    for i in range(img_num1):
        A[2*i] = np.hstack((XY_3D1[i],np.array([0,0,0,0])))
        A[2*i+1] = np.hstack((np.array([0,0,0,0]),XY_3D1[i]))
    for j in range(img_num2):
        A[2*img_num1+2*j] = np.hstack((XY_3D2[j],np.array([0,0,0,0])))
        A[2*img_num1+2*j+1] = np.hstack((np.array([0,0,0,0]),XY_3D2[j]))

    b = np.zeros((2*(img_num1+img_num2),1))
    for i in range(img_num1):
        b[2*i] = img1[i,0]
        b[2*i+1] = img1[i,1]
    for j in range(img_num2):
        b[2*img_num1+2*j] = img2[j,0]
        b[2*img_num1+2*j+1] = img2[j,1]

    P = np.dot(np.linalg.inv(np.dot(A.T,A)),(np.dot(A.T,b)))
    P = P.reshape(2,4)
    P = np.vstack((P, np.array([0,0,0,1])))

    return P

def compute_K_from_vanish_points(v1,v2,v3):
    """
    通过相互正交的平行线对计算出来的消影点，计算no-skew, square pixel的摄像机内参K
    此时，k的形式为：[[a,0,b],
                    [0,a,c],
                    [0,0,1]]
    :param v1:(x,y),非齐次坐标
    :param v2:(x,y),非齐次坐标
    :param v3:(x,y),非齐次坐标
    :return:
    """
    A= np.ones((3,4))
    A[0,:3] = np.array([v1[0]*v2[0]+v1[1]*v2[1],v1[0]+v2[0],v1[1]+v2[1]])
    A[1,:3] = np.array([v1[0]*v3[0]+v1[1]*v3[1],v1[0]+v3[0],v1[1]+v3[1]])
    A[2,:3] = np.array([v2[0]*v3[0]+v2[1]*v3[1],v2[0]+v3[0],v2[1]+v3[1]])
    U,D,VT = np.linalg.svd(A)
    w = VT[-1]
    omega = np.array([[w[0], 0, w[1]],
                      [0, w[0], w[2]],
                      [w[1],w[2],w[3]]])
    KT_inv = np.linalg.cholesky(omega)
    K = np.linalg.inv(KT_inv.T)
    K /= K[2,2]
    return K

def compute_rotation_matrix_between_cameras(v1,v2,v3,v4,v5,v6,K):
    """
    v1，v2,v3与v4,v5,v6分别是3D空间中的3对平行线在两张图片中对应的消影点。
    利用消影点计算对应平行线在各自摄像机坐标系下的方向，从而计算摄像机的旋转角度

    :param v1:(x,y),非齐次坐标
    :param v2:(x,y),非齐次坐标
    :param v3:(x,y),非齐次坐标
    :param v4:(x,y),非齐次坐标
    :param v5:(x,y),非齐次坐标
    :param v6:(x,y),非齐次坐标
    :param K:摄像机内参
    :return:旋转矩阵，3x3, 如果需要具体的角度需要从矩阵中再做处理
    """
    v1,v2,v3,v4,v5,v6 = np.hstack((v1,1)), np.hstack((v2,1)), np.hstack((v3,1)), np.hstack((v4,1)), np.hstack((v5,1)), np.hstack((v6,1)),

    d1 = np.linalg.inv(K).dot(v1)/np.linalg.norm(np.linalg.inv(K).dot(v1))
    d2 = np.linalg.inv(K).dot(v2)/np.linalg.norm(np.linalg.inv(K).dot(v2))
    d3 = np.linalg.inv(K).dot(v3)/np.linalg.norm(np.linalg.inv(K).dot(v3))
    d4 = np.linalg.inv(K).dot(v4)/np.linalg.norm(np.linalg.inv(K).dot(v4))
    d5 = np.linalg.inv(K).dot(v5)/np.linalg.norm(np.linalg.inv(K).dot(v5))
    d6 = np.linalg.inv(K).dot(v6)/np.linalg.norm(np.linalg.inv(K).dot(v6))

    D = np.zeros((3,3))
    D[:,0] = d1.T
    D[:,1] = d2.T
    D[:,2] = d3.T

    D_Prime = np.zeros((3,3))
    D_Prime[:,0] = d4.T
    D_Prime[:,1] = d5.T
    D_Prime[:,2] = d6.T

    R = np.linalg.inv(D_Prime).dot(D)
    return R






