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
        mean (float): Mean of gaussian noise. Default 128.
        sigma (float): Sigma of gaussian noise. Default 20.
    """

    def __init__(self, prob=0.5, mean=128, sigma=20):
        self.prob = prob
        self.mean = mean
        self.sigma = sigma

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            results["img"] = self._add_gaussian_noise(results["img"])
        return results

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        row, col, ch = img.shape
        gauss_noise = np.random.normal(self.mean, self.sigma, (row, col, ch))
        noisy_img = img + gauss_noise * 0.5
        return noisy_img
