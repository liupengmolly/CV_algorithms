import numpy as np

def get_homo_cor(X):
    """
    得到X的齐次坐标

    :param X: array(N, dimension)
    :return: array(N,dimension+1)
    """
    return np.hstack((X,np.ones((X.shape[0],1))))

def compute_rectified_image(im, H):
    new_x = np.zeros(im.shape[:2])
    new_y = np.zeros(im.shape[:2])
    for y in range(im.shape[0]): # height
        for x in range(im.shape[1]): # width
            new_location = H.dot([x, y, 1])
            new_location /= new_location[2]
            new_x[y,x] = new_location[0]
            new_y[y,x] = new_location[1]
    offsets = (new_x.min(), new_y.min())
    new_x -= offsets[0]
    new_y -= offsets[1]
    new_dims = (int(np.ceil(new_y.max()))+1,int(np.ceil(new_x.max()))+1)

    H_inv = np.linalg.inv(H)
    new_image = np.zeros(new_dims)

    for y in range(new_dims[0]):
        for x in range(new_dims[1]):
            old_location = H_inv.dot([x+offsets[0], y+offsets[1], 1])
            old_location /= old_location[2]
            old_x = int(old_location[0])
            old_y = int(old_location[1])
            if old_x >= 0 and old_x < im.shape[1] and old_y >= 0 and old_y < im.shape[0]:
                new_image[y,x] = im[old_y, old_x]

    return new_image, offsets


def reprojection_error(point_3d, image_points, camera_matrixs):
    """
    验证摄像机矩阵或者3D空间点坐标的准确性，计算各自的摄像机矩阵将3D空间中的点投影到各自摄像机平面
    的坐标，与所提供的image_points的误差。

    :param point_3d:
    :param image_points:
    :param camera_matrixs:
    :return: 返回所有对应摄像机下点的x,y坐标的偏差
    """
    N = image_points.shape[0]
    error_set = []
    points_3D_homo = np.hstack((point_3d, 1))

    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrixs[i]
        Yi = Mi.dot(points_3D_homo)

        pi_prime = np.array([Yi[0], Yi[1]])/Yi[2]
        error_i = (pi_prime-pi)
        error_set.append(error_i[0])
        error_set.append(error_i[1])

    error = np.array(error_set)
    return error

