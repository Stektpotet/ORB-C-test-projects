import argparse

import numpy as np
from timeit import Timer
from skimage import data, filters, color
from skimage.feature import orb
import matplotlib.pyplot as plt

detector: orb.ORB
img_gray: np.ndarray
img: np.ndarray


def _orb() -> None:
    detector.detect_and_extract(img_gray)


def _orb_c() -> None:
    detector.detect_and_extract_multichannel(img, img_gray)


def benchmark_image(image: np.ndarray, img_name: str, runs=100) -> (float, float):
    global detector
    global img
    global img_gray
    img = image.astype(np.float64)
    img_gray = color.rgb2gray(img)
    detector = orb.ORB()

    orb_time = Timer("""_orb()""", """from __main__ import _orb""").timeit(runs)
    orb_c_time = Timer("""_orb_c()""", """from __main__ import _orb_c""").timeit(runs)

    print(f"\t\t{img_name}\n"
          f"\tORB x100:\t\t{orb_time} s\n"
          f"\tORB-c x100:\t\t{orb_c_time} s\n\n"
          f"Slowdown ratio:\t{orb_c_time/orb_time}\n"
          f"\t\t\t\t\t{100*orb_c_time/orb_time:.3f}%")

    return orb_time, orb_c_time



if __name__ == '__main__':

    image_names = ["astronaut.png", "chelsea.png", "chessboard_RGB.png", "coffee.png", "color.png",
                   "hubble_deep_field.jpg", "ihc.png", "logo.png", "motorcycle_left.png", "motorcycle_right.png",
                   "retina.jpg", "rocket.jpg"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=[100], nargs=1)
    args = parser.parse_args()
    runs = args.runs[0]

    print(f"Running comparison {runs} times for all images in data-package...")
    images = [data._load(i) for i in image_names]

    orb_times = []
    orb_c_times = []
    slowdown_rate = []

    slowdown_cumulative = 0

    for i, img in enumerate(images):
        orb_time, orb_c_time = benchmark_image(img, image_names[i], runs)
        orb_times.append(orb_time)
        orb_c_times.append(orb_c_time)
        slowdown = orb_c_time/orb_time
        slowdown_rate.append(slowdown)
        slowdown_cumulative += slowdown

    slowdown_average = slowdown_cumulative / len(images)
    print(slowdown_average)
    print(f"{100*slowdown_average:.3f}%")

    x = np.arange(len(image_names))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, orb_times, width, label='ORB')
    rects2 = ax.bar(x + width / 2, orb_c_times, width, label='ORB-c')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (s)')
    ax.set_title('Speed comparison between ORB and ORB-c')
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    ax.set_xticklabels(image_names)
    ax.legend()

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    bars = ax.bar(x - width / 2, slowdown_rate, width, label='Slowdown')
    bars2 = ax.bar(x + width / 2, np.ones_like(slowdown_rate), width, label='Baseline')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Slowdown impact')
    ax.set_title('Speed impact of ORB-c compared to ORB')
    ax.set_xticks(x)
    ax.axhline(y=slowdown_average, linestyle='dashed', color='#279925')
    plt.xticks(rotation=90)
    ax.set_xticklabels(image_names)
    ax.legend()

    fig.tight_layout()

    plt.show()
