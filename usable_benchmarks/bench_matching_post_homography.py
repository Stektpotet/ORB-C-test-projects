import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
from skimage import data, filters, color
from skimage.feature import orb, match_descriptors, plot_matches
from skimage import transform as tf

from dataset import DataSet

features: orb.ORB
validity_distance: float


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _orb(img1: np.ndarray, img2: np.ndarray):
    matches, valid, invalid = _orb_match(
        features, img1, img2,
        lambda orb: orb.detect_and_extract(img1),
        lambda orb: orb.detect_and_extract(img2)
    )

    print(f"ORB\nNumber of matches: {matches}\n"
          f"Number of valid matches: {valid}\n"
          f"Number of invalid matches: {invalid}")

    return matches, valid, invalid, valid/matches, invalid/matches


def _orb_match(orb, img1, img2, f1, f2):
    f1(orb)
    desc1 = orb.descriptors
    keypoints1 = orb.keypoints

    f2(orb)
    desc2 = orb.descriptors
    keypoints2 = orb.keypoints

    matches = match_descriptors(desc1, desc2, cross_check=True)
    valid_matches = []
    invalid_matches = []

    for i in range(matches.shape[0]):
        src_point = keypoints1[matches[i, 0]]
        dst_point = keypoints2[matches[i, 1]]
        if dist(*src_point, *dst_point) <= validity_distance:
            valid_matches.append(matches[i])
        else:
            invalid_matches.append(matches[i])

    valid_matches = np.array(valid_matches)
    invalid_matches = np.array(invalid_matches)

    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))
    # plot_matches(ax[0],
    #              img1,
    #              img2,
    #              keypoints1, keypoints2, valid_matches)
    # ax[0].axis('off')
    # ax[0].set_title("Valid Matches")
    #
    # plot_matches(ax[1],
    #              img1,
    #              img2,
    #              keypoints1, keypoints2, invalid_matches)
    # ax[1].axis('off')
    # ax[1].set_title("Invalid Matches")
    #
    # point_correlation = np.stack((np.arange(len(keypoints1)), np.arange(len(keypoints1))), 1)
    # plot_matches(ax[2],
    #              img1,
    #              img2,
    #              keypoints1, keypoints1, point_correlation)
    # ax[2].axis('off')
    # ax[2].set_title("Ideal Correlation")
    #
    # plt.show()

    return len(matches), len(valid_matches), len(invalid_matches)

def _orb_c(img1: np.ndarray, img1_gray: np.ndarray, img2: np.ndarray, img2_gray: np.ndarray):
    """
    :param img1:
    :param img1_gray:
    :param img2:
    :param img2_gray:
    :return:
        match_count, valid_match_count, invalid_match_count,
        normalized_valid_match_count, normalized_invalid_match_count
    """
    matches, valid, invalid = _orb_match(
        features, img1, img2,
        lambda orb: orb.detect_and_extract_multichannel(img1, img1_gray),
        lambda orb: orb.detect_and_extract_multichannel(img2, img2_gray)
    )

    print(f"ORB-c\nNumber of matches: {matches}\n"
          f"Number of valid matches: {valid}\n"
          f"Number of invalid matches: {invalid}")

    return matches, valid, invalid, valid/matches, invalid/matches


def bench_valid_matches(img: np.ndarray, img2: np.ndarray, trs: np.ndarray):
    img_gray = color.rgb2gray(img)
    img2 = tf.warp(img2, trs)
    img2_gray = color.rgb2gray(img2)
    # TODO: Return a structure with names
    return [_orb(img_gray, img2_gray), _orb_c(img, img_gray, img2, img2_gray)]
    # return [_orb_c(img, img_gray, img2, img2_gray)]


if __name__ == '__main__':
    validity_distance = 20

    # variants = [
    #     (0.8, 0.1, 0.1),
    #     (0.1, 0.8, 0.1),
    #     (0.1, 0.1, 0.8)
    # ]
    #
    # for v in variants:
    #     bench_prefix = f"r{v[0]*100:.0f}g{v[1]*100:.0f}b{v[2]*100:.0f}_"
    #     features = orb.ORB(channel_weights=v)
    #     ds = DataSet("Color sensitive", ["../dataset/bark/", "../dataset/trees/"])
    #     ds.benchmark_plot(bench_valid_matches, bench_prefix)

    bench_prefix = "AND_"

    features = orb.ORB()
    # features = orb.ORB(n_keypoints=2)

    datasets =[
        DataSet("Viewpoint Warp", ["../dataset/graffiti/", "../dataset/wall/"]),
        DataSet("Blur", ["../dataset/bikes/", "../dataset/trees/"]),
        DataSet("Zoom And Rotation", ["../dataset/bark/"]),  # boat is grayscale
        DataSet("Light", ["../dataset/leuven/"]),
        DataSet("JPEG Compression", ["../dataset/ubc/"])
    ]
    for ds in datasets:
        ds.benchmark_plot(bench_valid_matches, bench_prefix)

