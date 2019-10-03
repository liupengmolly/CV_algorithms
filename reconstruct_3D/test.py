import numpy as np
import scipy.io as sio
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera.camera import Camera
from reconstruct_3D.space_carving import form_initial_voxels,get_voxel_bounds,carve

def axis_equal(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_surface(voxels, voxel_size = 0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # First grid the data
    res = np.amax(voxels[1,:] - voxels[0,:])
    ux = np.unique(voxels[:,0])
    uy = np.unique(voxels[:,1])
    uz = np.unique(voxels[:,2])

    # Expand the model by one step in each direction
    ux = np.hstack((ux[0] - res, ux, ux[-1] + res))
    uy = np.hstack((uy[0] - res, uy, uy[-1] + res))
    uz = np.hstack((uz[0] - res, uz, uz[-1] + res))

    # Convert to a grid
    X, Y, Z = np.meshgrid(ux, uy, uz)

    # Create an empty voxel grid, then fill in the elements in voxels
    V = np.zeros(X.shape)
    N = voxels.shape[0]
    for ii in range(N):
        ix = ux == voxels[ii,0]
        iy = uy == voxels[ii,1]
        iz = uz == voxels[ii,2]
        V[iy, ix, iz] = 1

    marching_cubes = measure.marching_cubes(V, 0, spacing=(voxel_size, voxel_size, voxel_size))
    verts = marching_cubes[0]
    faces = marching_cubes[1]
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=0, color='red')
    axis_equal(ax, verts[:, 0], verts[:,1], verts[:,2])
    plt.show()

def estimate_silhouette(im):
    """
    因为图像数据是红色的，所有简单选取用红色通道大于其他两个通道的像素作为阴影轮廓
    :param im:
    :return:
    """
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = True
    frames = sio.loadmat('../data/ps3/space_carving/frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            print(i)
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    if use_true_silhouette:
        plot_surface(voxels)

    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)  
    plot_surface(voxels, voxel_size)
