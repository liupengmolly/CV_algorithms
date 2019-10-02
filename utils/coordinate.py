import numpy as np

def get_homo_cor(X):
    """
    得到X的齐次坐标

    :param X: array(N, dimension)
    :return: array(N,dimension+1)
    """
    return np.hstack((X,np.ones((X.shape[0],1))))
