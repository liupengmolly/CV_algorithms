import numpy as np

def get_data_from_txt_file(filename, use_subset = False):
    with open(filename) as f:
        lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i+1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i,2] = z
        points[i,0] = x
        points[i,1] = y
    return points
