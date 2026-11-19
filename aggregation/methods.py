import numpy as np
from typing import List

class Aggregator:
    @staticmethod
    def fed_avg(weights_list: List[np.ndarray]) -> np.ndarray:
        """w_new = (1/n) * sum(w_i)"""
        return np.mean(weights_list, axis=0)

    @staticmethod
    def multi_krum_indices(weights_list: List[np.ndarray], f: int) -> List[int]:
        """Indices of the n-f clients Multi-Krum keeps (lowest Krum score)."""
        weights = np.array(weights_list)
        n = len(weights_list)
        k = max(1, n - f - 2)

        scores = []
        for i in range(n):
            dists = np.square(np.linalg.norm(weights - weights[i], axis=1))
            dists_sorted = np.sort(dists)
            scores.append(np.sum(dists_sorted[1:k + 1]))  # exclude self (distance 0)

        return np.argsort(scores)[:max(1, n - f)].tolist()

    @staticmethod
    def multi_krum(weights_list: List[np.ndarray], f: int) -> np.ndarray:
        """
        score_i = sum over k-NN(i) of ||w_i - w_j||_2^2,  k = n - f - 2
        keep the n-f lowest-score clients, average them.
        (Blanchard et al., 2017)
        """
        weights = np.array(weights_list)
        kept_indices = Aggregator.multi_krum_indices(weights_list, f)
        return np.mean(weights[kept_indices], axis=0)

    @staticmethod
    def trimmed_mean(weights_list: List[np.ndarray], beta: float = 0.2) -> np.ndarray:
        """Extra standard baseline (not in the core formula set): coordinate-wise,
        drop top/bottom beta fraction, average the rest."""
        weights = np.array(weights_list)
        n = len(weights_list)
        k = int(beta * n)
        if k == 0:
            return np.mean(weights, axis=0)
        sorted_weights = np.sort(weights, axis=0)
        return np.mean(sorted_weights[k:n - k], axis=0)

    @staticmethod
    def median(weights_list: List[np.ndarray]) -> np.ndarray:
        """Extra standard baseline (not in the core formula set): coordinate-wise median."""
        return np.median(weights_list, axis=0)

    @staticmethod
    def strategy_a_kept(n: int, flagged_indices: List[int]) -> List[int]:
        """Hospitals that stay in the average (everyone not flagged; all if all flagged)."""
        kept = [i for i in range(n) if i not in flagged_indices]
        return kept if kept else list(range(n))

    @staticmethod
    def strategy_a_hard_exclusion(weights_list: List[np.ndarray], flagged_indices: List[int]) -> np.ndarray:
        """w_new = (1/|kept|) * sum_{i not in flagged} w_i"""
        weights = np.array(weights_list)
        kept_indices = Aggregator.strategy_a_kept(len(weights_list), flagged_indices)
        return np.mean(weights[kept_indices], axis=0)

    @staticmethod
    def strategy_b_weights(z_scores: np.ndarray, lambda_param: float = 0.5) -> np.ndarray:
        """weight_i = exp(-lambda * max(0, z_i)), normalized to sum to 1."""
        raw_weights = np.exp(-lambda_param * np.maximum(0, z_scores))
        return raw_weights / np.sum(raw_weights)

    @staticmethod
    def strategy_b_soft_weighting(weights_list: List[np.ndarray], z_scores: np.ndarray, lambda_param: float = 0.5) -> np.ndarray:
        """
        weight_i = exp(-lambda * max(0, z_i))
        w_tilde_i = weight_i / sum_j(weight_j)
        w_new = sum_i w_tilde_i * w_i
        """
        weights = np.array(weights_list)
        norm_weights = Aggregator.strategy_b_weights(z_scores, lambda_param)
        return np.sum(weights * norm_weights[:, np.newaxis], axis=0)
