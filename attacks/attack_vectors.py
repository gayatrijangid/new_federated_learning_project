"""
attack_vectors.py
==================
10 attack types, each implemented to match its real published mechanism
(not a placeholder weight-nudge). Organized by what they actually touch:

  UPDATE-LEVEL (operate on the trained weight delta):
    gradient_scaling, sign_flipping, free_rider, gaussian_noise, byzantine

  COORDINATED (need visibility into what OTHER hospitals are doing this round):
    sybil

  DATA-LEVEL (operate on X/y before local training -- the honest trainer
  then trains on poisoned data, exactly like a real backdoored client would):
    clean_label_backdoor, dirty_label_backdoor, label_flipping

  PASSIVE / SIDE-CHANNEL (does NOT touch the submitted update at all --
  an eavesdropper tries to reconstruct data from an honestly-trained
  gradient; scored separately, never affects model accuracy):
    gradient_inversion_probe
"""
import numpy as np


class AttackEngine:
    GRADIENT_SCALE_S = 10.0
    FREE_RIDER_SIGMA = 0.5
    GAUSSIAN_NOISE_SIGMA = 1.0
    BYZANTINE_RANGE = 50.0

    BACKDOOR_TRIGGER_FEATURE_IDX = 4   # "Insulin" column in the standardized Pima feature set
    BACKDOOR_TRIGGER_VALUE = 6.0       # extreme standardized value, well outside normal range
    BACKDOOR_TARGET_CLASS = 0          # attacker wants triggered inputs classified as "non-diabetic"

    # ------------------------------------------------------------------
    # UPDATE-LEVEL ATTACKS -- operate on (w_honest, w_global)
    # ------------------------------------------------------------------
    @classmethod
    def gradient_scaling(cls, w_honest: np.ndarray, w_global: np.ndarray, s: float = None) -> np.ndarray:
        """delta = w_honest - w_global ; w' = w_global + s*delta  (boosting/train-and-scale attack)"""
        s = cls.GRADIENT_SCALE_S if s is None else s
        delta = w_honest - w_global
        return w_global + s * delta

    @classmethod
    def sign_flipping(cls, w_honest: np.ndarray, w_global: np.ndarray) -> np.ndarray:
        """delta = w_honest - w_global ; w' = w_global - delta  (reflect the honest update)"""
        delta = w_honest - w_global
        return w_global - delta

    @classmethod
    def free_rider(cls, w_global: np.ndarray, sigma: float = None) -> np.ndarray:
        """Hospital does not train at all -- submits global weights + noise, free-riding
        off everyone else's work."""
        sigma = cls.FREE_RIDER_SIGMA if sigma is None else sigma
        return w_global + np.random.normal(0, sigma, size=w_global.shape)

    @classmethod
    def gaussian_noise(cls, w_honest: np.ndarray, sigma: float = None) -> np.ndarray:
        """Generic Byzantine-style corruption: honestly-trained update corrupted with noise."""
        sigma = cls.GAUSSIAN_NOISE_SIGMA if sigma is None else sigma
        return w_honest + np.random.normal(0, sigma, size=w_honest.shape)

    @classmethod
    def byzantine(cls, w_global: np.ndarray) -> np.ndarray:
        """True arbitrary/unstructured Byzantine behavior (Blanchard et al. 2017 / Yin et al.
        2018 definition): a fresh, unpredictable extreme vector every round -- the
        worst-case a Byzantine-robust aggregator is supposed to tolerate."""
        return np.random.uniform(-cls.BYZANTINE_RANGE, cls.BYZANTINE_RANGE, size=w_global.shape)

    # ------------------------------------------------------------------
    # DATA-LEVEL ATTACKS -- operate on (X, y) BEFORE local training
    # ------------------------------------------------------------------
    @classmethod
    def label_flipping(cls, y: np.ndarray) -> np.ndarray:
        return 1 - y

    @classmethod
    def dirty_label_backdoor(cls, X: np.ndarray, y: np.ndarray) -> tuple:
        """Standard (dirty-label) backdoor: insert the trigger into ALL local samples
        and FORCE their label to the target class, regardless of true label. Stronger
        and more reliable than clean-label, but the poisoned labels are visibly
        'wrong' if anyone inspected the raw data."""
        X_bd = X.copy()
        X_bd[:, cls.BACKDOOR_TRIGGER_FEATURE_IDX] = cls.BACKDOOR_TRIGGER_VALUE
        y_bd = np.full_like(y, cls.BACKDOOR_TARGET_CLASS)
        return X_bd, y_bd

    @classmethod
    def clean_label_backdoor(cls, X: np.ndarray, y: np.ndarray) -> tuple:
        """Clean-label backdoor: insert the trigger ONLY into samples that already
        belong to the target class, so every label stays truthful ('clean'). Stealthier
        (harder to catch by inspecting data) but less reliable, since the model has to
        pick up the trigger-target association without any mislabeled examples."""
        X_bd = X.copy()
        target_mask = (y == cls.BACKDOOR_TARGET_CLASS)
        X_bd[target_mask, cls.BACKDOOR_TRIGGER_FEATURE_IDX] = cls.BACKDOOR_TRIGGER_VALUE
        return X_bd, y  # labels untouched -- this is what makes it "clean"

    @staticmethod
    def make_triggered_test_set(X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """Builds the held-out evaluation set used to measure Attack Success Rate:
        every NON-target-class test sample gets the trigger inserted. ASR = fraction
        of these (truly non-target) samples the global model now predicts as the
        target class BECAUSE of the trigger."""
        non_target_mask = (y_test != AttackEngine.BACKDOOR_TARGET_CLASS)
        X_trig = X_test[non_target_mask].copy()
        X_trig[:, AttackEngine.BACKDOOR_TRIGGER_FEATURE_IDX] = AttackEngine.BACKDOOR_TRIGGER_VALUE
        return X_trig, non_target_mask.sum()

    # ------------------------------------------------------------------
    # COORDINATED ATTACK -- needs the full per-round assignment, not just
    # one hospital's own data. Called once per round by simulation.py with
    # the list of hospital indices that are "sybil" this round.
    # ------------------------------------------------------------------
    @classmethod
    def sybil_coordinated_payload(cls, sybil_members_honest_updates: list, w_global: np.ndarray, boost: float = 3.0) -> np.ndarray:
        """ALL hospitals flagged 'sybil' this round submit the SAME coordinated
        update -- this identical-update collusion is the actual defining trait of a
        Sybil attack (and exactly what similarity-based Sybil defenses like
        FoolsGold (Fung et al., 2020) are built to catch). We build the shared
        payload from the mean of the sybil group's own honest gradients, boosted,
        so it still points in a learnable (if poisoned-strength) direction rather
        than pure noise."""
        mean_honest = np.mean(sybil_members_honest_updates, axis=0)
        delta = mean_honest - w_global
        return w_global + boost * delta

    # ------------------------------------------------------------------
    # PASSIVE PRIVACY PROBE -- does NOT touch the submitted update. Simulates
    # an eavesdropper trying to reconstruct the hospital's data FROM an
    # honestly-trained gradient (simplified gradient-matching / mini-DLG for
    # logistic regression). Returns a leakage score; never affects accuracy.
    # ------------------------------------------------------------------
    @staticmethod
    def gradient_inversion_probe(X_true: np.ndarray, y_true: np.ndarray,
                                   w_global: np.ndarray, w_honest: np.ndarray,
                                   learning_rate: float, epochs: int,
                                   steps: int = 50, lr_probe: float = 0.5) -> dict:
        """Observes the real gradient (w_honest - w_global), then tries to find a
        SYNTHETIC x_hat whose own gradient best matches it -- a simplified
        single-sample Deep-Leakage-from-Gradients (Zhu et al., 2019) style attack
        adapted to logistic regression. Returns how close the reconstruction got
        to the TRUE average input (which the probe itself never sees -- only used
        here, by us, to SCORE the leakage after the fact)."""
        observed_delta = w_honest - w_global
        n_features = X_true.shape[1]

        x_hat = np.random.randn(n_features) * 0.1
        y_hat_label = float(np.mean(y_true) >= 0.5)

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

        for _ in range(steps):
            x_b = np.append(x_hat, 1.0)
            pred = sigmoid(x_b @ w_global)
            synthetic_grad = (pred - y_hat_label) * x_b
            synthetic_delta = -learning_rate * epochs * synthetic_grad[:n_features]

            # gradient of ||synthetic_delta - observed_delta||^2 w.r.t. x_hat (finite-difference-free,
            # using the fact that synthetic_delta is approx linear in x_hat for small steps)
            residual = synthetic_delta - observed_delta[:n_features]
            x_hat -= lr_probe * residual

        true_mean_x = np.mean(X_true, axis=0)
        reconstruction_similarity = float(np.dot(x_hat, true_mean_x) /
                                           (np.linalg.norm(x_hat) * np.linalg.norm(true_mean_x) + 1e-8))
        reconstruction_error = float(np.linalg.norm(x_hat - true_mean_x))

        return {
            "reconstruction_similarity": round(reconstruction_similarity, 4),
            "reconstruction_error": round(reconstruction_error, 4),
        }
