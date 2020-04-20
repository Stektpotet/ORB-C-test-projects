import matplotlib.pyplot as plt
from typing import List
import numpy as np


class DataSubset:
    def __init__(self, root, img_count: int = 6):
        self.base_image = f"{root}img1.ppm"
        self.images = [f"{root}img{i}.ppm" for i in range(2, img_count + 1)]
        self.homographies = [f"{root}H1to{i}p" for i in range(2, img_count + 1)]
        self.image_count = img_count

    def benchmark(self, bench_func):
        results = []
        img1 = plt.imread(self.base_image)
        for i in range(len(self.images)):
            img2 = plt.imread(self.images[i])
            m = np.loadtxt(self.homographies[i])
            results.append(bench_func(img1, img2, m))
        return results


class DataSet:
    def __init__(self, name, paths: List[str]):
        self.name = name
        self.subsets = [DataSubset(path) for path in paths]

    def benchmark(self, bench_func):
        results = []
        for subset in self.subsets:
            results.append(subset.benchmark(bench_func))
        return results

    def benchmark_plot(self, bench_func):

        def autolabel(rects, totals):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for i, rect in enumerate(rects):
                total = totals[i]
                base = rect.get_y()
                value = rect.get_height()
                label_y = - value / 2 if base < 0 else value / 2
                if value != 0:
                    ax.annotate(f'{value * 100 / total:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, label_y),
                                xytext=(0, -12),  # -6 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                    if abs(value) > 24:  # if there's space for both labels
                        ax.annotate(f'{value}',
                                    xy=(rect.get_x() + rect.get_width() / 2, label_y),
                                    xytext=(0, 0),  # points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')

        width = 0.45
        padding = 10

        for i, subset in enumerate(self.subsets):
            subset_result = np.array(subset.benchmark(bench_func))
            orb_results = subset_result[:, 0]
            orb_c_results = subset_result[:, 1]

            y_max = max(np.amax(orb_results[:, 1]), np.amax(orb_c_results[:, 1])) + padding
            y_min = -max(np.amax(orb_results[:, 2]), np.amax(orb_c_results[:, 2])) - padding

            fig, ax = plt.subplots()
            x = np.arange(subset.image_count - 1)  # all images except base image

            autolabel(
                ax.bar(x - width / 2, orb_results[:, 2], width, bottom=orb_results[:, 2] * -1, label='ORB Invalid'),
                orb_results[:, 0]
            )
            autolabel(
                ax.bar(x - width / 2, orb_results[:, 1], width, label='ORB Valid'),
                orb_results[:, 0]
            )
            autolabel(
                ax.bar(x + width / 2, orb_c_results[:, 2], width, bottom=orb_c_results[:, 2] * -1,
                       label='ORB-c Invalid'),
                orb_c_results[:, 0]
            )
            autolabel(
                ax.bar(x + width / 2, orb_c_results[:, 1], width, label='ORB-c Valid'),
                orb_c_results[:, 0]
            )

            ax.set_ylim(y_min, y_max)

            ax.yaxis.set_visible(False)
            ax.set_ylabel('Matches')
            ax.set_xlabel('Image variant')
            ax.set_title(f'{self.name}[#{i}] Matches ORB vs ORB-c')
            ax.legend()
            plt.savefig(f"../plots/{self.name}_{i}.png")
            plt.show()
