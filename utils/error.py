import numpy as np

def reproj_mse(camera_matrix,XY_3D1,XY_3D2,img1,img2):
    """
    通过计算出的摄像机矩阵，将3D空间中的点重新投影回成像平面的最小均方误差

    :param camera_matrix: 计算出的摄像机矩阵,(3,4)
    :param XY_3D1:真实3D空间中深度为Z1的齐次坐标,(img_num1,4)
    :param XY_3D2:真实3D空间中深度为Z2的齐次坐标,(img_num2,4)
    :param img1: 物体在深度为Z1时的成像2D齐次坐标，(img_num1,3)
    :param img2: 物体在深度为Z2时的成像2D齐次坐标，(img_num2,3)

    :return: 计算得到的误差
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

    camera_P = camera_matrix[:2,:].reshape(8,1)
    b_reproj = np.dot(A,camera_P)
    error = np.sum((b_reproj-b)**2)
    error /= (img_num1+img_num2)

    return error
