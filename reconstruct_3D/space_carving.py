import numpy as np

def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    """
    创建用于雕刻的体素的基本网格

    :param xlim:
    :param ylim:
    :param zlim:
    :param num_voxels:
    :return:
    """
    x_dim = xlim[-1] - xlim[0]
    y_dim = ylim[-1] - ylim[0]
    z_dim = zlim[-1] - zlim[0]

    total_volume = x_dim * y_dim * z_dim
    voxel_volume = float(total_volume /num_voxels)
    voxel_size = np.cbrt(voxel_volume)

    x_voxel_num = np.round(x_dim /voxel_size)
    y_voxel_num = np.round(y_dim /voxel_size)
    z_voxel_num = np.round(z_dim /voxel_size)

    x_coor = np.linspace(xlim[0]+0.5*voxel_size, xlim[0]+(0.5+x_voxel_num-1)*voxel_size, x_voxel_num)
    y_coor = np.linspace(ylim[0]+0.5*voxel_size, ylim[0]+(0.5+y_voxel_num-1)*voxel_size, y_voxel_num)
    z_coor = np.linspace(zlim[0]+0.5*voxel_size, zlim[0]+(0.5+z_voxel_num-1)*voxel_size, z_voxel_num)

    XX, YY, ZZ = np.meshgrid(x_coor, y_coor, z_coor)
    voxels = np.vstack((XX.reshape(-1),YY.reshape(-1),ZZ.reshape(-1))).T

    return voxels, voxel_size

def get_voxel_bounds(cameras,estimate_better_bounds=False, num_voxels=4000):
    """
    计算一个用于雕刻物体的立方体的边界

    :param cameras: camera对象
   :param estimate_better_bounds:
    :param num_voxels:
    :return:
    """
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:, 0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:, 1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:, 2].min(), camera_positions[:,2].max()]

    # 计算摄像机朝向的zlim
    camera_range = 0.6 * np.sqrt((xlim[1]-xlim[0])**2 + (ylim[1]-ylim[0])**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min(zlim[0], viewpoint[2])
        zlim[1] = max(zlim[1], viewpoint[2])

    # 因为物体一定在圈内，将边界稍微缩小
    xlim = xlim + (xlim[1]-xlim[0]) / 4 * np.array([1,-1])
    ylim = ylim + (ylim[1]-ylim[0]) / 4 * np.array([1,-1])

    if estimate_better_bounds:
        voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)
        for c in cameras:
            voxels = carve(voxels, c)

        xlim = [voxels[0][0]-1.5*voxel_size, voxels[0][0]+1.5*voxel_size]
        ylim = [voxels[0][1]-1.5*voxel_size, voxels[0][1]+1.5*voxel_size]
        zlim = [voxels[0][2]-1.5*voxel_size, voxels[0][2]+1.5*voxel_size]

    return xlim,ylim,zlim

def carve(voxels, camera):
    """
    对于给定的voxels, 根据camera对象提供的信息进行雕刻

    :param voxels:
    :param camera:
    :return:
    """
    homo_voxels = np.hstack((voxels, np.ones((voxels.shape[0], 1)))).T

    # 记录体素下标
    N = voxels.shape[0]
    voxel_index = np.arange(0,N)

    P = camera.P
    img_voxels = P.dot(homo_voxels)
    img_voxels /= img_voxels[2,:]
    img_voxels = img_voxels[:2, :].T

    # 检查投影后的体素是否在图像范围内
    img_y_max, img_x_max = camera.silhouette.shape
    img_y_min, img_x_min = 0, 0

    voxelX = img_voxels[:,0]
    x_range_filter = np.all([voxelX>img_x_min, voxelX<img_x_max], axis=0)
    img_voxels = img_voxels[x_range_filter, :]
    voxel_index = voxel_index[x_range_filter]

    voxelY = img_voxels[:,1]
    y_range_filter = np.all([voxelY>img_y_min, voxelY<img_y_max], axis=0)
    img_voxels = img_voxels[y_range_filter, :]
    voxel_index = voxel_index[y_range_filter]

    # 检查该体素是否在图像对应的阴影范围内
    img_voxels = img_voxels.astype(int)
    silhouette_filter = (camera.silhouette[img_voxels[:,1], img_voxels[:,0]]==1)
    voxel_index = voxel_index[silhouette_filter]

    return voxels[voxel_index,:]




