import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
from skimage import data, filters, color
from skimage.feature import orb, match_descriptors, plot_matches
from skimage import transform as tf

features: orb.ORB
validity_distance: float


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _orb(img1: np.ndarray, img2: np.ndarray, trs: np.ndarray):
    features.detect_and_extract(img1)
    desc1 = features.descriptors
    keypoints1 = features.keypoints

    # keypoints1 = np.array([np.array([400-i*10, 0+i*20]) for i in range(20)])

    features.detect_and_extract(img2)
    desc2 = features.descriptors
    keypoints2 = features.keypoints

    matches = match_descriptors(desc1, desc2, cross_check=True)
    print(matches)
    transformed_src_points = tf.matrix_transform(keypoints1, trs)

    valid_matches = []
    invalid_matches = []

    for i in range(matches.shape[0]):
        src_point = transformed_src_points[matches[i, 0]]
        dst_point = keypoints2[matches[i, 1]]
        if dist(*src_point, *dst_point) <= validity_distance:
            valid_matches.append(matches[i])
        else:
            invalid_matches.append(matches[i])

    valid_matches = np.array(valid_matches)
    invalid_matches = np.array(invalid_matches)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    plot_matches(ax[0],
                 img1,
                 img2,
                 keypoints1, keypoints2, valid_matches)
    ax[0].axis('off')
    ax[0].set_title("Valid Matches")

    # Show where the keypoints SHOULD HAVE ended up
    point_correlation = np.stack((np.arange(len(keypoints1)), np.arange(len(keypoints1))), axis=1)
    plot_matches(ax[1],
                 img1,
                 img2,
                 keypoints1, transformed_src_points, point_correlation)
    ax[1].axis('off')
    ax[1].set_title("Invalid Matches")

    plt.show()
    print(f"ORB\nNumber of matches: {matches.shape[0]}\n"
          f"Number of valid matches: {valid_matches.shape[0]}\n"
          f"Number of invalid matches: {invalid_matches.shape[0]}")

# For reference
# TODO: https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html#sphx-glr-auto-examples-transform-plot-matching-py

def _orb_c(img1: np.ndarray, img1_gray: np.ndarray, img2: np.ndarray, img2_gray: np.ndarray, trs: np.ndarray):
    features.detect_and_extract_multichannel(img1, img1_gray)
    desc1 = features.descriptors
    keypoints1 = features.keypoints

    # keypoints1 = np.array([np.array([400-i*10, 0+i*20]) for i in range(20)])

    features.detect_and_extract_multichannel(img2, img2_gray)
    desc2 = features.descriptors
    keypoints2 = features.keypoints

    matches = match_descriptors(desc1, desc2, cross_check=True)

    transformed_src_points = tf.matrix_transform(coords=keypoints2, matrix=trs)

    valid_matches = []
    invalid_matches = []

    for i in range(matches.shape[0]):
        src_point = transformed_src_points[matches[i, 0]]
        dst_point = keypoints1[matches[i, 1]]
        if dist(*src_point, *dst_point) <= validity_distance:
            valid_matches.append(matches[i])
        else:
            invalid_matches.append(matches[i])

    valid_matches = np.array(valid_matches)
    invalid_matches = np.array(invalid_matches)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    plot_matches(ax[0],
                 img1,
                 img2,
                 keypoints1, keypoints2, valid_matches)
    ax[0].axis('off')
    ax[0].set_title("Valid Matches")

    img1_to_img2 = tf.warp(img1, tf.AffineTransform(trs).inverse)
    img2_to_img1 = tf.warp(img2, tf.AffineTransform(trs))

    # Show where the keypoints SHOULD HAVE ended up
    point_correlation = np.stack((np.arange(len(keypoints1)), np.arange(len(keypoints1))), 1)
    plot_matches(ax[1],
                 img2_to_img1,
                 img1,
                 transformed_src_points, keypoints2, point_correlation)
    ax[1].axis('off')
    ax[1].set_title("CORRELATION")

    plt.show()
    print(f"ORB-c\nNumber of matches: {matches.shape[0]}\n"
          f"Number of valid matches: {valid_matches.shape[0]}\n"
          f"Number of invalid matches: {invalid_matches.shape[0]}")


def bench_valid_matches(img: np.ndarray, img2: np.ndarray, trs: np.ndarray):
    img_gray = color.rgb2gray(img)
    img2_gray = color.rgb2gray(img2)
    # TODO: Return number of valid matching elements - Then Compare the between ORB and ORB-c
    # _orb(img_gray, img2_gray, trs)
    _orb_c(img, img_gray, img2, img2_gray, trs)


if __name__ == '__main__':
    validity_distance = 10

    path = "../dataset/bikes/"
    compare_nr = 2

    img = plt.imread(f"{path}img1.ppm")
    img2 = plt.imread(f"{path}img{compare_nr}.ppm")

    transform = np.loadtxt(f"{path}H1to{compare_nr}p")

    features = orb.ORB(n_keypoints=10)

    bench_valid_matches(img, img2, transform)
