import matplotlib.pyplot as plt
from typing import List
import numpy as np


class DataSubset:
    def __init__(self, root, img_count: int = 6):
        self.name = root[root.rfind('/', 0, -1) + 1:root.rfind('/')]
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

    def benchmark_plot(self, bench_func, bench_prefix=""):
        VALID = 1
        INVALID = 2
        NRM_VALID = 3
        NRM_INVALID = 4

        def autolabel(rects, counts):
            plt.rcParams.update({'font.size': 8})
            """Attach a text label above each bar in *rects*, displaying its height."""
            for i, rect in enumerate(rects):
                count = counts[i]
                base = rect.get_y()
                value = rect.get_height()
                label_y = - value / 2 if base < 0 else value / 2
                if value != 0:
                    ax.annotate(f'{value:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, label_y),
                                xytext=(0, -10),  # -6 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                    if abs(value) > 18:  # if there's space for both labels
                        ax.annotate(f'{count:.0f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, label_y),
                                    xytext=(0, -1),  # points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
            plt.rcParams.update({'font.size': 12})

        width = 0.47

        for i, subset in enumerate(self.subsets):
            subset_result = np.array(subset.benchmark(bench_func))
            orb_results = subset_result[:, 0]
            orb_c_results = subset_result[:, 1]

            fig, ax = plt.subplots()
            x = np.arange(subset.image_count - 1)  # all images except base image


            autolabel(
                ax.bar(x - width / 2,
                       orb_results[:, NRM_VALID] * 100,
                       width,
                       label='ORB Valid',
                       color='#ff6a00'),
                orb_results[:, VALID]
            )
            autolabel(
                ax.bar(x - width / 2,
                       orb_results[:, NRM_INVALID]*100,
                       width,
                       bottom=orb_results[:, NRM_INVALID] * -100,
                       label='ORB Invalid',
                       color='#006dba'),
                orb_results[:, INVALID]
            )
            autolabel(
                ax.bar(x + width / 2,
                       orb_c_results[:, NRM_VALID]*100,
                       width,
                       label='ORB-c Valid',
                       color='#279925'),
                orb_c_results[:, VALID]
            )
            autolabel(
                ax.bar(x + width / 2,
                       orb_c_results[:, NRM_INVALID]*100,
                       width,
                       bottom=orb_c_results[:, NRM_INVALID] * -100,
                       label='ORB-c Invalid', color='#972599'),
                orb_c_results[:, INVALID]
            )

            ax.axhline(color='k', linestyle='dashed')
            ax.set_ylim(-100, 100)

            ax.yaxis.set_visible(False)
            ax.set_ylabel('Matches')
            ax.set_xlabel('Image variant')
            ax.set_title(f'{self.name}[#{i} - {subset.name}] Matches ORB vs ORB-c {f"({bench_prefix})" if bench_prefix else ""}')
            ax.legend()
            plt.savefig(f"../plots/{bench_prefix}{self.name}_{i}_{subset.name}.png")
            plt.show()
