import matplotlib.pyplot as plt
from typing import List
import numpy as np


class DataSubset:
    def __init__(self, root, img_count: int = 6):
        self.base_image = f"{root}img1.ppm"
        self.images = [f"{root}img{i}.ppm" for i in range(2, img_count + 1)]
        self.homographies = [f"{root}H1to{i}p" for i in range(2, img_count + 1)]
        self.image_count = img_count

class DataSet:

    def __init__(self, name, paths: List[str]):
        self.name = name
        self.subsets = [DataSubset(path) for path in paths]

    def benchmark(self, bench_func):
        results = []

        for subset in self.subsets:
            img1 = plt.imread(subset.base_image)
            for i in range(len(subset.images)):
                img2 = plt.imread(subset.images[i])
                m = np.loadtxt(subset.homographies[i])
                results.append(bench_func(img1, img2, m))

        return results
