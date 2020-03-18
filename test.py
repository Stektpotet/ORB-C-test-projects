import numpy as np
from matplotlib import pyplot as plt
from skimage import transform as tf
from skimage import data, filters, color, exposure
from skimage.feature import orb, match_descriptors, plot_matches, corner_harris, corner_orientations, corner_peaks, corner_fast
import random

from skimage.feature import orb_cy
def prepare(im: np.ndarray):
    return filters.gaussian(color.rgb2gray(im), sigma=0.5)


if __name__ == '__main__':
    #
    # with open("orb-c_color_indices.txt", mode="w") as file:
    #     for _ in range(255):
    #         file.write(str(random.randint(0, 2)))
    #         file.write(' ')
    #     file.write(str(random.randint(0, 2)))

    # Setup and prepare the image to work on
    img = data.astronaut().astype(np.float64)



    # img2 = tf.warp(img, tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
    #                            translation=(0, -200)))
    img2 = tf.warp(img, tf.ProjectiveTransform())
    img2 = tf.warp(img, tf.AffineTransform(scale=(2, 2), rotation=0.5, translation=(-200, -400)))
    img_gray = color.rgb2gray(img)
    img_gray2 = color.rgb2gray(img2)


    detector = orb.ORB(n_keypoints=100)
    # detector.detect(img_gray)
    # detector.extract(img_gray, detector.keypoints, detector.scales, detector.orientations)
    # detector.extract_multichannel(img, detector.keypoints, detector.scales, detector.orientations)
    # detector.detect_and_extract(img_gray)
    detector.detect_and_extract_multichannel(img, img_gray)

    detector2 = orb.ORB(n_keypoints=100)
    # detector2.detect(img_gray2)
    # detector2.extract(img_gray2, detector2.keypoints, detector2.scales, detector2.orientations)
    # detector2.extract_multichannel(img2, detector2.keypoints, detector2.scales, detector2.orientations)
    # detector2.detect_and_extract(img_gray2)
    detector2.detect_and_extract_multichannel(img2, img_gray2)

    matches = match_descriptors(detector.descriptors, detector2.descriptors, cross_check=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax,
                 img.astype(np.uint8),
                 img2.astype(np.uint8),
                 detector.keypoints, detector2.keypoints, matches)
    ax.axis('off')
    ax.set_title("Original Image vs. Transformed Image")

    plt.show()