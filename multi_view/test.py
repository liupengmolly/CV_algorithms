import matplotlib.pyplot as plt
from scipy.misc import imread
from utils.io import get_data_from_txt_file
from utils.coordinate import compute_rectified_image
from utils.visual import plot_epipolar_lines_on_images
from multi_view.epipolar import lls_eight_points_alg,normalized_eight_points_alg,\
    compute_distance_to_epipolar_lines,compute_epipole,compute_matching_homographies

if __name__ == '__main__':
    im1 = imread('../data/set1/image1.jpg')
    im2 = imread('../data/set1/image2.jpg')

    points1 = get_data_from_txt_file('../data/set1/pt_2D_1.txt')
    points2 = get_data_from_txt_file('../data/set1/pt_2D_2.txt')

    F = lls_eight_points_alg(points1,points2)
    print(F)
    dis = compute_distance_to_epipolar_lines(points1,points2,F)
    print(dis)

    F = normalized_eight_points_alg(points1,points2)
    print(F)
    dis = compute_distance_to_epipolar_lines(points1,points2,F)
    print(dis)

    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print()
    print("H2:\n", H2)

    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_points_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()

