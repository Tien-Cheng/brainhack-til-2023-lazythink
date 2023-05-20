import random

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomGaussian(BaseTransform):
    """Add random gaussian noise to image

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, prob=0.5, mean=0, var=0.1):
        self.prob = prob
        self.mean = mean
        self.var = var
        self.sigma = self.var**0.5

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            results["img"] = self._add_gaussian_noise(results["img"])
        return results

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        row, col, ch = img.shape
        gauss_noise = np.random.normal(self.mean, self.sigma, (row, col, ch))
        noisy_img = img + gauss_noise
        return noisy_img
