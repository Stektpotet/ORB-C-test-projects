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

def homography_transform(points, H):
    N = points.shape[0]
    ones = np.ones((N, 1))
    points = np.append(points, ones, axis=1)

    transform_coordinates = np.zeros((points.shape[0],2))
    src_points = points.copy()

    #Important. APplying homograpy, the coordinates to be multiplied are [x, y, 1]. Imread returns [height x width / y, x I believe]!!!!
    i,j = 0,1
    for row in range(points.shape[0]):
        #Swap points in order to do the transformation [s*x', s*y', s]^T = H * [x, y, 1] correctly!!
        src_points[row][i], src_points[row][j] = src_points[row][j], src_points[row][i]
        #Find homogenous coordinates. Remember to scale!
        temp = H.dot(np.transpose(src_points[row]))
        scale = temp[-1]
        temp = temp/scale

        #Swap back so that it is consistent with the already-found keypoints!
        temp[i], temp[j] = temp[j], temp[i]
        transform_coordinates[row] = temp[:2]

    return transform_coordinates


def _orb(img1: np.ndarray, img2: np.ndarray, homography):

    matches, valid, invalid = _orb_match(
        features, img1, img2, homography,
        lambda orb: orb.detect_and_extract(img1),
        lambda orb: orb.detect_and_extract(img2)
    )

    print(f"ORB\nNumber of matches: {matches}\n"
          f"Number of valid matches: {valid}\n"
          f"Number of invalid matches: {invalid}")

    return matches, valid, invalid, valid/matches, invalid/matches


def _orb_c(img1: np.ndarray, img1_gray: np.ndarray, img2: np.ndarray, img2_gray: np.ndarray, homography):

    matches, valid, invalid = _orb_match(
        features, img1, img2, homography,
        lambda orb: orb.detect_and_extract_multichannel(img1, img1_gray),
        lambda orb: orb.detect_and_extract_multichannel(img2, img2_gray)
    )

    print(f"ORB-c\nNumber of matches: {matches}\n"
          f"Number of valid matches: {valid}\n"
          f"Number of invalid matches: {invalid}")

    return matches, valid, invalid, valid/matches, invalid/matches



def _orb_match(orb, img1, img2, homography, f1, f2):
    f1(orb)
    desc1 = orb.descriptors
    keypoints1 = orb.keypoints

    f2(orb)
    desc2 = orb.descriptors
    keypoints2 = orb.keypoints

    matches = match_descriptors(desc1, desc2, cross_check=True)
    valid_matches = []
    invalid_matches = []

    transformed_src_points = homography_transform(keypoints1, homography)

    for i in range(matches.shape[0]):
        src_point = transformed_src_points[matches[i, 0]]
        dst_point = keypoints2[matches[i, 1]]
        if dist(*src_point, *dst_point) <= validity_distance:
            valid_matches.append(matches[i])
        else:
            invalid_matches.append(matches[i])

    valid_matches = np.array(valid_matches)
    invalid_matches = np.array(invalid_matches)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))
    plot_matches(ax[0],
                 img1,
                 img2,
                 keypoints1, keypoints2, valid_matches)
    ax[0].axis('off')
    ax[0].set_title("Valid Matches")

    plot_matches(ax[1],
                 img1,
                 img2,
                 keypoints1, keypoints2, invalid_matches)
    ax[1].axis('off')
    ax[1].set_title("Invalid Matches")

    point_correlation = np.stack((np.arange(len(keypoints1)), np.arange(len(keypoints1))), 1)
    plot_matches(ax[2],
                 img1,
                 img2,
                 keypoints1, transformed_src_points, point_correlation)
    ax[2].axis('off')
    ax[2].set_title("Ideal Correlation")

    plt.show()

    return len(matches), len(valid_matches), len(invalid_matches)


def bench_valid_matches(img: np.ndarray, img2: np.ndarray, homography: np.ndarray):
    img_gray = color.rgb2gray(img)
    # img2 = tf.warp(img2, homography)
    img2_gray = color.rgb2gray(img2)
    # TODO: Return a structure with names
    return [_orb(img_gray, img2_gray, homography), _orb_c(img, img_gray, img2, img2_gray, homography)]
    # return [_orb_c(img, img_gray, img2, img2_gray)]

def bench_channel_weighting():
    global features
    variants = [
        (0.1, 0.1, 0.8),
        (0.1, 0.2, 0.7),
        (0.2, 0.1, 0.7),
        (0.1, 0.3, 0.6),
        (0.2, 0.2, 0.6),
        (0.3, 0.1, 0.6),
        (0.1, 0.4, 0.5),
        (0.2, 0.3, 0.5),
        (0.3, 0.2, 0.5),
        (0.4, 0.1, 0.5),
        (0.1, 0.5, 0.4),
        (0.2, 0.4, 0.4),
        (0.3, 0.3, 0.4),
        (0.4, 0.2, 0.4),
        (0.5, 0.1, 0.4),
        (0.1, 0.6, 0.3),
        (0.2, 0.5, 0.3),
        (0.3, 0.4, 0.3),
        (0.4, 0.3, 0.3),
        (0.5, 0.2, 0.3),
        (0.6, 0.1, 0.3),
        (0.1, 0.7, 0.2),
        (0.2, 0.6, 0.2),
        (0.3, 0.5, 0.2),
        (0.4, 0.4, 0.2),
        (0.5, 0.3, 0.2),
        (0.6, 0.2, 0.2),
        (0.7, 0.1, 0.2),
        (0.1, 0.8, 0.1),
        (0.2, 0.7, 0.1),
        (0.3, 0.6, 0.1),
        (0.4, 0.5, 0.1),
        (0.5, 0.4, 0.1),
        (0.6, 0.3, 0.1),
        (0.7, 0.2, 0.1),
        (0.8, 0.1, 0.1),
    ]

    datasets = [
        DataSet("Viewpoint Warp", ["../dataset/graffiti/", "../dataset/wall/"]),
        DataSet("Blur", ["../dataset/bikes/", "../dataset/trees/"]),
        DataSet("Zoom And Rotation", ["../dataset/bark/"]),  # boat is grayscale
        DataSet("Light", ["../dataset/leuven/"]),
        DataSet("JPEG Compression", ["../dataset/ubc/"])
    ]

    if len(variants) > 10:
        print("Benchmarking A LOT of variants - this WILL take a while!!!")

    for ds in datasets:
        for v in variants:
            bench_prefix = f"r{v[0]*100:.0f}g{v[1]*100:.0f}b{v[2]*100:.0f}_"
            features = orb.ORB(channel_weights=v)

            ds.benchmark_plot(bench_valid_matches, bench_prefix, "channel_search/")

def bench_uniform():
    global features

    datasets = [
        DataSet("Viewpoint Warp", ["../dataset/graffiti/", "../dataset/wall/"]),
        DataSet("Blur", ["../dataset/bikes/", "../dataset/trees/"]),
        DataSet("Zoom And Rotation", ["../dataset/bark/"]),  # boat is grayscale
        DataSet("Light", ["../dataset/leuven/"]),
        DataSet("JPEG Compression", ["../dataset/ubc/"])
    ]
    bench_prefix = "uniform_"  #f"r{v[0]*100:.0f}g{v[1]*100:.0f}b{v[2]*100:.0f}_"
    features = orb.ORB(n_keypoints=1000)

    for ds in datasets:
        ds.benchmark_plot(bench_valid_matches, bench_prefix)


if __name__ == '__main__':
    validity_distance = 20

    # These are the two benchmarking functions
    bench_uniform()             # Benchmark performance uniformly

    # This has no effect if ORB-c uses another patch descriptor test (Variant 2 or 3)
    bench_channel_weighting()   # Benchmark with channel weighting

    # To benchmark the other variants of ORB-c patch descriptor test
    # You have to go into the scikit-image skimage/feature/orb_cy.pyx
    # and swap out the active binary descriptor test
    # the three we've evaluated are present there - the unused ones are commented out
    #
    #
    # Additional benchmarks can be constructed as such:
    #
    # features = orb.ORB(channel_weights=(0.8, 0.1, 0.1), channel_weight_seed=42)
    # datasets =[
    #     DataSet("Viewpoint Warp", ["../dataset/graffiti/", "../dataset/wall/"]),
    #     DataSet("Blur", ["../dataset/bikes/", "../dataset/trees/"]),
    #     DataSet("Zoom And Rotation", ["../dataset/bark/"]),  # boat is grayscale
    #     DataSet("Light", ["../dataset/leuven/"]),
    #     DataSet("JPEG Compression", ["../dataset/ubc/"])
    # ]
    #
    # for ds in datasets:
    #     ds.benchmark_plot(bench_valid_matches, bench_prefix="bench01", plot_path="some_benchmark/")
