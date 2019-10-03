import numpy as np
from utils.coordinate import get_homo_cor
from utils.error import reproj_mse
from camera.calibrate import calibrate_affine_camera_matrix,compute_K_from_vanish_points
from single_view.single_view import compute_vanish_points,compute_angle_between_planes

def test_calibrate_affine_camera_matrix():
    real_XY = np.load('../data/real_XY.npy')
    front_image = np.load('../data/front_image.npy')
    back_image =np.load('../data/back_image.npy')

    img1, img2 = get_homo_cor(front_image), get_homo_cor(back_image)
    real_XY1 = np.hstack((real_XY,np.zeros((real_XY.shape[0],1))))
    real_XY2 = np.hstack((real_XY,np.ones((real_XY.shape[0],1))*150))
    XY1, XY2 = get_homo_cor(real_XY1),get_homo_cor(real_XY2)

    P = calibrate_affine_camera_matrix(XY1,XY2,img1,img2)
    print(P)

    error = reproj_mse(P,XY1,XY2,img1,img2)
    print(error)

def test_calibrate_from_vanish_points():
    pa1 = np.array([674,1826])
    pa2 = np.array([2456,1060])
    pb1 = np.array([1094,1340])
    pb2 = np.array([1774,1086])

    pc1 = pa1
    pc2 = np.array([126,1056])
    pd1 = pa2
    pd2 = np.array([1940,866])

    pe1 = pb1
    pe2 = np.array([1080,598])
    pf1 = np.array([504,900])
    pf2 = np.array([424,356])

    pg1 = np.array([1774,1086])
    pg2 = np.array([1840,478])

    v1 = compute_vanish_points(pa1,pa2,pb1,pb2)
    v2 = compute_vanish_points(pc1,pc2,pd1,pd2)
    v3 = compute_vanish_points(pe1,pe2,pf1,pf2)
    v4 = compute_vanish_points(pe1,pe2,pg1,pg2)

    print(v1,v2,v3)
    K = compute_K_from_vanish_points(v1,v2,v3)
    print(K)

    theta = compute_angle_between_planes(v1,v2,v3,v4,K)
    print(theta)

if __name__ == '__main__':
    test_calibrate_from_vanish_points()
