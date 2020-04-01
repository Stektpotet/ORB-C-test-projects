import numpy as np
import math
from timeit import Timer
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage import data, filters, color
from skimage.feature import orb, match_descriptors, plot_matches

features: orb.ORB
validity_distance: float


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _orb(img1, img2, trs: tf.AffineTransform):
    features.detect_and_extract(img1)
    desc1 = features.descriptors
    keypoints1 = features.keypoints

    features.detect_and_extract(img2)
    desc2 = features.descriptors
    keypoints2 = features.keypoints

    matches = match_descriptors(desc1, desc2, cross_check=True)

    # TODO: try transpose of the matrix
    transformed_src_points = tf.matrix_transform(keypoints1[:], trs.params)


    valid_matches = []

    for i in range(matches.shape[0]):
        src_point = transformed_src_points[matches[i, 0]]
        dst_point = keypoints2[matches[i, 1]]
        if dist(*src_point, *dst_point) <= validity_distance:
            valid_matches.append(matches[i])

    valid_matches = np.array(valid_matches)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    plot_matches(ax[0],
                 img1.astype(np.uint8),
                 img2.astype(np.uint8),
                 keypoints1, keypoints2, valid_matches)
    ax[0].axis('off')
    ax[1].set_title("Original Image vs. Transformed Image")

    point_correlation = np.stack((np.arange(len(keypoints1)), np.arange(len(keypoints1))), 1)
    plot_matches(ax[1],
                 img1.astype(np.uint8),
                 img2.astype(np.uint8),
                 keypoints1, transformed_src_points, point_correlation)
    ax[1].axis('off')
    ax[1].set_title("Point transformation Correlation")

    plt.show()
    print(f"Number of matches: {matches.shape[0]}\tNumber of keypoints: {features.n_keypoints}")


def _orb_c(img1, img2, img1_gray, img2_gray, trs: tf.AffineTransform):
    features.detect_and_extract_multichannel(img1, img1_gray)
    desc1 = features.descriptors
    features.detect_and_extract_multichannel(img2, img2_gray)
    desc2 = features.descriptors

    matches = match_descriptors(desc1, desc2, cross_check=True)
    print(f"Number of matches: {matches.shape[0]}\tNumber of keypoints: {features.n_keypoints}")


def bench_valid_matches(img: np.ndarray, trs: tf.AffineTransform):
    img_transformed = tf.warp(img, trs)

    img_gray = color.rgb2gray(img)
    img_transformed_gray = tf.warp(img_gray, trs)

    # TODO: Return number of valid matching elements - Then Compare the between ORB and ORB-c
    _orb(img_gray, img_transformed_gray, trs)
    # _orb_c(img, img_transformed, img_gray, img_transformed_gray, trs)


if __name__ == '__main__':
    validity_distance = 100
    img = data.stereo_motorcycle()[0].astype(np.float64)
    features = orb.ORB(n_keypoints=20)

    transform = tf.AffineTransform(scale=(1.2, 1.5), rotation=0.65, translation=(200, -260))

    bench_valid_matches(img, transform)

