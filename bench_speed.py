import numpy as np
from timeit import Timer
from skimage import data, filters, color
from skimage.feature import orb

detector: orb.ORB
img_gray: np.ndarray
img: np.ndarray


def _orb():
    detector.detect_and_extract(img_gray)


def _orb_c():
    detector.detect_and_extract_multichannel(img, img_gray)


if __name__ == '__main__':

    img = data.astronaut().astype(np.float64)
    img_gray = color.rgb2gray(img)
    detector = orb.ORB()

    timer_orb = Timer("""_orb()""", """from __main__ import _orb""")
    timer_orb_c = Timer("""_orb_c()""", """from __main__ import _orb_c""")
    print(timer_orb.timeit(100))
    print(timer_orb_c.timeit(100))
