import numpy as np
import math

def compute_vanish_points(pa1,pa2,pb1,pb2):
    """
    通过图像中两条平行线上的点计算平行线的消影点
    :param pa1:
    :param pa2:
    :param pb1:
    :param pb2:
    :return: 消影点的坐标
    """
    m1 = (pa2[1]-pa1[1])/(pa2[0]-pa1[0])
    m2 = (pb2[1]-pb1[1])/(pb2[0]-pb1[0])
    b1 = pa2[1]-pa2[0]*m1
    b2 = pb2[1]-pb2[0]*m2

    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    v = np.array([x,y])
    return v

def compute_angle_between_planes(v1,v2,v3,v4,K):
    """
    通过平面一中的消影点v1，v2和平面2中的消影点v3，v4计算两个平面
    在3D空间中的夹角

    :param v1:(x,y),非齐次坐标
    :param v2:(x,y),非齐次坐标
    :param v3:(x,y),非齐次坐标
    :param v4:(x,y),非齐次坐标
    :return: 夹角theta
    """

    omega_inv = K.dot(K.T)
    v1,v2,v3,v4 = np.hstack((v1,1)),np.hstack((v2,1)),np.hstack((v3,1)),np.hstack((v4,1))
    l1 = np.cross(v1.T,v2.T)
    l2 = np.cross(v3.T,v4.T)

    cos_theta = (l1.T.dot(omega_inv).dot(l2))/(np.sqrt(l1.T.dot(omega_inv).dot(l1))*np.sqrt(l2.T.dot(omega_inv).dot(l2)))
    theta = np.arccos(cos_theta)*180/math.pi

    return theta