import numpy as np

import matplotlib.pyplot as plt
from skimage import data, filters, color
from skimage.feature import orb, match_descriptors, plot_matches

detector: orb.ORB

# https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html#sphx-glr-auto-examples-transform-plot-fundamental-matrix-py


if __name__ == '__main__':
    img1, img2, _ = data.stereo_motorcycle()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img1_gray = color.rgb2gray(img1)
    img2_gray = color.rgb2gray(img2)

    detector = orb.ORB(n_keypoints=100)
    detector.detect_and_extract_multichannel(img1, img1_gray)
    img1_desc = detector.descriptors
    img1_keys = detector.keypoints

    detector.detect_and_extract_multichannel(img2, img2_gray)
    img2_desc = detector.descriptors
    img2_keys = detector.keypoints

    matches = match_descriptors(img1_desc, img2_desc, cross_check=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax,
                 img1.astype(np.uint8),
                 img2.astype(np.uint8),
                 img1_keys, img2_keys, matches)

    ax.axis('off')
    ax.set_title("Original Image vs. Transformed Image")

    plt.show()