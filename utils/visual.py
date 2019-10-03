import matplotlib.pyplot as plt

def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    plt.subplot(1,2,1)
    ln1 = F.T.dot(points2.T)
    for i in range(ln1.shape[1]):
        plt.plot([0, im1.shape[1]], [-ln1[2][i]*1.0/ln1[1][i], -(ln1[2][i]+ln1[0][i]*im1.shape[1])*1.0/ln1[1][i]], 'r')
        plt.plot([points1[i][0]], [points1[i][1]], 'b*')
    plt.imshow(im1, cmap='gray')

    plt.subplot(1,2,2)
    ln2 = F.dot(points1.T)
    for i in range(ln2.shape[1]):
        plt.plot([0, im2.shape[1]], [-ln2[2][i]*1.0/ln2[1][i], -(ln2[2][i]+ln2[0][i]*im2.shape[1])/ln2[1][i]], 'r')
        plt.plot([points2[i][0]], [points2[i][1]], 'b*')
    plt.imshow(im2, cmap='gray')

