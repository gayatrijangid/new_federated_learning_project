import numpy as np
from typing import Tuple, List

class DistanceDetector:
    """
    w_bar = mean update across all clients
    d_i   = || w_i - w_bar ||_2
    median = med(d_1..d_n)
    MAD    = med(|d_i - median|)
    z_i    = 0.6745 * (d_i - median) / (MAD + epsilon)
    flagged if z_i > tau   (tau = 2.5 by default)
    """
    def __init__(self, tau: float = 2.5, epsilon: float = 1e-8):
        self.tau = tau
        self.epsilon = epsilon

    def detect(self, weights_list: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        weights = np.array(weights_list)

        w_mean = np.mean(weights, axis=0)
        distances = np.linalg.norm(weights - w_mean, axis=1)

        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))

        z_scores = 0.6745 * (distances - median_dist) / (mad + self.epsilon)
        flagged_indices = np.where(z_scores > self.tau)[0].tolist()

        return z_scores, flagged_indices
