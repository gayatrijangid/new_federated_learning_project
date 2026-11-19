"""
simulation.py
==============
Core federated-learning simulation loop. No FastAPI/pydantic dependency,
so it can be run standalone (test_run.py) or imported by main.py (the API).
"""
import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from federated.local_model import LocalLogisticRegression
from detection.distance_detector import DistanceDetector
from aggregation.methods import Aggregator
from attacks.attack_vectors import AttackEngine
from evaluate.metrics import Evaluator

VALID_METHODS = ["FedAvg", "Multi-Krum", "Trimmed-Mean", "Median",
                  "System A (Hard Exclusion)", "System B (Soft Weighting)"]

# All 10 requested attacks.
UPDATE_LEVEL_ATTACKS = {"gradient_scaling", "sign_flipping", "free_rider", "gaussian_noise", "byzantine"}
DATA_LEVEL_ATTACKS = {"label_flipping", "clean_label_backdoor", "dirty_label_backdoor"}
COORDINATED_ATTACKS = {"sybil"}
PASSIVE_ATTACKS = {"gradient_inversion_probe"}

VALID_ATTACKS = sorted(UPDATE_LEVEL_ATTACKS | DATA_LEVEL_ATTACKS | COORDINATED_ATTACKS | PASSIVE_ATTACKS)


DATASETS = {
    "pima": {
        "label": "Pima Indians Diabetes",
        "task": "diabetes",
        "positive": "diabetic",
        "patients": 768, "features": 8,
        "split_feature": 1, "split_name": "Glucose",
        "note": "768 patients, 8 clinical features, diabetic vs not diabetic",
    },
    "breast_cancer": {
        "label": "Breast Cancer Wisconsin (Diagnostic)",
        "task": "breast cancer",
        "positive": "malignant",
        "patients": 569, "features": 30,
        "split_feature": 0, "split_name": "mean radius",
        "note": "569 patients, 30 tumour measurements, malignant vs benign",
    },
}
DEFAULT_DATASET = "pima"


def _read_dataset(name: str):
    """Returns (X, y) with y = 1 for the disease/positive class."""
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer      # bundled with scikit-learn, no download
        d = load_breast_cancer()
        return d.data, (1 - d.target).astype(int)            # sklearn: 0 = malignant -> flip so malignant = 1
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv"), names=cols)
    return df.drop(columns=["Outcome"]).values, df["Outcome"].values


def load_dataset(name: str, n_hospitals: int, seed: int = 42):
    """Loads one of the real clinical datasets and splits the training part
    non-IID across hospitals: patients are sorted by one key measurement
    (Glucose for Pima, mean radius for Breast Cancer), so every hospital sees a
    different patient population, like real hospitals."""
    if name not in DATASETS:
        name = DEFAULT_DATASET
    X, y = _read_dataset(name)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    sort_idx = np.argsort(X_train[:, DATASETS[name]["split_feature"]])
    X_sorted, y_sorted = X_train[sort_idx], y_train[sort_idx]
    hospital_X = np.array_split(X_sorted, n_hospitals)
    hospital_y = np.array_split(y_sorted, n_hospitals)

    return list(zip(hospital_X, hospital_y)), X_test, y_test


def load_real_pima_dataset(n_hospitals: int, seed: int = 42):
    """Kept for backward compatibility (test_run.py)."""
    return load_dataset("pima", n_hospitals, seed)


MIN_VARIETY = 20   # use the "not side by side" rule only if at least this many arrangements exist


def ring_placements(n: int, k: int) -> int:
    """How many ways to choose k of n hospitals with none side by side
    (H1 and Hn count as neighbours on the ring)."""
    if k == 0:
        return 1
    if k > n // 2:
        return 0
    return n * math.comb(n - k, k) // (n - k)


def random_malicious_indices(n: int, k: int, rng=None):
    """Pick which hospitals are malicious AT RANDOM (a different set every call).

    When there is plenty of choice, no two malicious hospitals sit side by side
    (H1 and Hn count as neighbours on the ring in the network view), and every
    such placement is equally likely. When "not side by side" would leave fewer
    than MIN_VARIETY arrangements (for example 5 of 10, which allows only 2), the
    hospitals are simply picked fully at random so the choice really changes.
    (index.html has the same algorithm in randomPlacement().)"""
    rng = rng or random.Random()
    k = max(0, min(n - 1, int(k)))
    if k == 0:
        return []
    if ring_placements(n, k) >= MIN_VARIETY:
        extra = n - 2 * k                       # honest hospitals left over after one gap each
        cuts = sorted(rng.sample(range(extra + k - 1), k - 1))
        bins, prev = [], -1
        for c in cuts + [extra + k - 1]:        # split the extra honest hospitals among the k gaps
            bins.append(c - prev - 1)
            prev = c
        pos, out = rng.randrange(n), []
        for i in range(k):
            out.append(pos % n)
            pos += 2 + bins[i]
        return sorted(out)
    return sorted(rng.sample(range(n), k))


def _clean(x, lo=-1000.0, hi=1000.0, nd=3):
    """JSON-safe float: no NaN/Infinity (they would break the browser's JSON.parse)."""
    return round(float(np.clip(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo), lo, hi)), nd)


def _influence_weights(method, n, flagged, z_scores, client_weights, num_malicious):
    """How much each hospital actually contributes to the new global model this
    round. Defined for the methods that aggregate by hospital; Median and
    Trimmed-Mean work coordinate-by-coordinate, so they have no per-hospital
    weight (returns None -> shown as n/a in the UI)."""
    if method == "FedAvg":
        return [round(1.0 / n, 6)] * n
    if method == "Multi-Krum":
        kept = Aggregator.multi_krum_indices(client_weights, f=num_malicious)
        return [round(1.0 / len(kept), 6) if i in kept else 0.0 for i in range(n)]
    if method == "System A (Hard Exclusion)":
        kept = Aggregator.strategy_a_kept(n, flagged)
        return [round(1.0 / len(kept), 6) if i in kept else 0.0 for i in range(n)]
    if method == "System B (Soft Weighting)":
        return [round(float(w), 6) for w in Aggregator.strategy_b_weights(z_scores)]
    return None


def run_simulation_stream(rounds: int, num_hospitals: int, num_malicious: int,
                           selected_methods, attack_types, dataset: str = DEFAULT_DATASET,
                           malicious_ids=None, attack_provider=None):
    """Generator: runs the federated simulation and YIELDS one event per round
    so a UI can show it live.

      {"type": "start", ...}   config, which hospitals are malicious, methods
      {"type": "round", ...}   per round: attack each malicious hospital used,
                               and for every defense: accuracy, flagged hospitals,
                               z-scores and influence weights
      {"type": "delivery", ...} the best model, sent back to every hospital
      {"type": "final", ...}   final classification + detection metrics

    attack_provider (optional): a function round -> {hospital_id: attack_name}. When given,
    the attacks come from it every round (this is how a person on another computer attacks
    live) and the set of malicious hospitals can change from round to round.
    """
    num_hospitals = int(max(3, min(40, num_hospitals)))
    rounds = int(max(1, min(200, rounds)))

    dataset = dataset if dataset in DATASETS else DEFAULT_DATASET
    hospital_data, X_test, y_test = load_dataset(dataset, num_hospitals)

    n_features = hospital_data[0][0].shape[1]
    n_weights = n_features + 1
    # Which hospitals are malicious: the ids sent by the page (so what you saw in the
    # preview is what runs), otherwise a fresh random choice. Chosen ONCE per run, so every
    # defense in the run faces exactly the same attackers.
    chosen = None
    if malicious_ids is not None:
        try:
            chosen = sorted({int(i) for i in malicious_ids if 0 <= int(i) < num_hospitals})[:num_hospitals - 1]
        except (TypeError, ValueError):
            chosen = None
    if chosen is None:
        chosen = random_malicious_indices(num_hospitals, num_malicious)
    malicious_indices = chosen
    num_malicious = len(malicious_indices)
    live = attack_provider is not None
    if live:
        malicious_indices, num_malicious = [], 0
    ever_attacked = set()

    trainer = LocalLogisticRegression(learning_rate=0.1, epochs=5)
    detector = DistanceDetector(tau=2.5)

    methods = [m for m in selected_methods if m in VALID_METHODS]
    attacks = [a for a in attack_types if a in VALID_ATTACKS] or ["sign_flipping"]

    trajectories = {method: [] for method in methods}
    detection_evaluators = {method: Evaluator() for method in methods}
    global_weights = {method: np.zeros(n_weights) for method in methods}
    classification_metrics = {}
    privacy_leakage = {method: [] for method in methods}  # gradient_inversion_probe scores

    yield {
        "type": "start",
        "config": {"dataset": dataset, "rounds": rounds, "num_hospitals": num_hospitals,
                   "num_malicious": num_malicious, "malicious_ids": malicious_indices, "selected_methods": methods,
                   "attack_types": attacks, "live": live},
        "malicious_ids": malicious_indices,
        "methods": methods,
    }

    for r in range(1, rounds + 1):
        # The attack each malicious hospital runs this round is the same for
        # every defense (it only depends on the hospital and the round).
        if live:
            attack_assignment = {}
            for h, a in (attack_provider(r) or {}).items():
                try:
                    h = int(h)
                except (TypeError, ValueError):
                    continue
                if 0 <= h < num_hospitals and a in VALID_ATTACKS:
                    attack_assignment[h] = a
            malicious_now = sorted(attack_assignment)
        else:
            attack_assignment = {i: attacks[(rank + r) % len(attacks)] for rank, i in enumerate(malicious_indices)}
            malicious_now = malicious_indices
        ever_attacked.update(malicious_now)
        round_methods = {}

        for method in methods:
            w_global = global_weights[method]

            # ---- Step 2: every hospital trains locally (on poisoned data if a
            #      data-level attack is assigned; honestly otherwise) ----
            honest_trained = {}
            for i in range(num_hospitals):
                X_h, y_h = hospital_data[i]
                attack_type = attack_assignment.get(i, "none")

                if attack_type in DATA_LEVEL_ATTACKS:
                    if attack_type == "label_flipping":
                        y_h = AttackEngine.label_flipping(y_h)
                    elif attack_type == "clean_label_backdoor":
                        X_h, y_h = AttackEngine.clean_label_backdoor(X_h, y_h)
                    elif attack_type == "dirty_label_backdoor":
                        X_h, y_h = AttackEngine.dirty_label_backdoor(X_h, y_h)

                honest_trained[i] = trainer.train(X_h, y_h, w_global)

            # ---- Step 3: coordinate Sybil group (identical shared payload) ----
            sybil_members = [i for i, a in attack_assignment.items() if a == "sybil"]
            sybil_payload = None
            if sybil_members:
                sybil_payload = AttackEngine.sybil_coordinated_payload(
                    [honest_trained[i] for i in sybil_members], w_global)

            # ---- Step 4: apply update-level / coordinated attacks, build submissions ----
            client_weights = []
            for i in range(num_hospitals):
                attack_type = attack_assignment.get(i, "none")
                w_honest = honest_trained[i]

                if attack_type == "gradient_scaling":
                    w_submit = AttackEngine.gradient_scaling(w_honest, w_global)
                elif attack_type == "sign_flipping":
                    w_submit = AttackEngine.sign_flipping(w_honest, w_global)
                elif attack_type == "free_rider":
                    w_submit = AttackEngine.free_rider(w_global)
                elif attack_type == "gaussian_noise":
                    w_submit = AttackEngine.gaussian_noise(w_honest)
                elif attack_type == "byzantine":
                    w_submit = AttackEngine.byzantine(w_global)
                elif attack_type == "sybil":
                    w_submit = sybil_payload
                elif attack_type == "gradient_inversion_probe":
                    w_submit = w_honest  # passive: submission itself is honest
                    X_h, y_h = hospital_data[i]
                    leak = AttackEngine.gradient_inversion_probe(
                        X_h, y_h, w_global, w_honest, trainer.learning_rate, trainer.epochs)
                    privacy_leakage[method].append({"round": r, "hospital": i, **leak})
                else:
                    w_submit = w_honest  # clean-label/dirty-label/label-flip already trained-in above, or honest

                client_weights.append(w_submit)

            # ---- Step 5: detection + aggregation (unchanged) ----
            z_scores, flagged = detector.detect(client_weights)
            detection_evaluators[method].update_detection_stats(malicious_now, flagged, num_hospitals)

            if method == "FedAvg":
                w_next = Aggregator.fed_avg(client_weights)
            elif method == "Multi-Krum":
                w_next = Aggregator.multi_krum(client_weights, f=len(malicious_now))
            elif method == "Trimmed-Mean":
                w_next = Aggregator.trimmed_mean(client_weights)
            elif method == "Median":
                w_next = Aggregator.median(client_weights)
            elif method == "System A (Hard Exclusion)":
                w_next = Aggregator.strategy_a_hard_exclusion(client_weights, flagged)
            elif method == "System B (Soft Weighting)":
                w_next = Aggregator.strategy_b_soft_weighting(client_weights, z_scores)
            else:
                w_next = Aggregator.fed_avg(client_weights)

            global_weights[method] = w_next
            y_pred = trainer.predict(X_test, w_next)
            acc = Evaluator.compute_accuracy(y_test, y_pred)
            trajectories[method].append(acc)

            if r == rounds:
                classification_metrics[method] = Evaluator.compute_classification_metrics(y_test, y_pred)

            round_methods[method] = {
                "accuracy": acc,
                "flagged": [int(i) for i in flagged],
                "z": [_clean(z) for z in z_scores],
                "weights": _influence_weights(method, num_hospitals, flagged, z_scores,
                                              client_weights, len(malicious_now)),
            }

        best_now = max(round_methods, key=lambda m: round_methods[m]["accuracy"])
        yield {
            "type": "round",
            "round": r,
            "attacks": {str(i): a for i, a in attack_assignment.items()},
            "malicious_ids": malicious_now,
            "methods": round_methods,
            "best": {"method": best_now, "accuracy": round_methods[best_now]["accuracy"]},
        }

    # ---- the best model goes back to every hospital ----
    best_method = max(methods, key=lambda m: trajectories[m][-1])
    w_best = global_weights[best_method]
    delivered_acc = trajectories[best_method][-1]
    hospitals_out = []
    for i in range(num_hospitals):
        X_h, y_h = hospital_data[i]
        w_alone = np.zeros(n_weights)                     # what this hospital could do by itself
        for _ in range(rounds):
            w_alone = trainer.train(X_h, y_h, w_alone)
        alone_acc = Evaluator.compute_accuracy(y_test, trainer.predict(X_test, w_alone))
        hospitals_out.append({"id": i, "attacked": i in ever_attacked,
                              "alone_acc": alone_acc, "delivered_acc": delivered_acc})
    yield {
        "type": "delivery",
        "best_method": best_method,
        "accuracy": delivered_acc,
        "rounds": rounds,
        "hospitals": hospitals_out,
        "model": {"dataset": dataset, "method": best_method, "bias_is_last": True,
                  "weights": [round(float(w), 6) for w in w_best]},
    }

    metrics_resp = {}
    for method in methods:
        det = detection_evaluators[method].get_detection_metrics()
        entry = {
            "classification": classification_metrics[method],
            "detection": det,
            "variance": round(float(np.std(trajectories[method][-10:])), 2),
        }
        if privacy_leakage[method]:
            last_round_leaks = [p for p in privacy_leakage[method] if p["round"] == rounds]
            entry["gradient_inversion_leakage"] = last_round_leaks
        metrics_resp[method] = entry

    yield {"type": "final", "metrics": metrics_resp}


def run_simulation(rounds: int, num_hospitals: int, num_malicious: int,
                    selected_methods, attack_types, dataset: str = DEFAULT_DATASET,
                    malicious_ids=None):
    """Non-streaming version (used by test_run.py and POST /api/simulate):
    runs everything and returns (trajectories, metrics) at the end."""
    trajectories, metrics = {}, None
    for ev in run_simulation_stream(rounds, num_hospitals, num_malicious,
                                    selected_methods, attack_types, dataset, malicious_ids):
        if ev["type"] == "start":
            trajectories = {m: [] for m in ev["methods"]}
        elif ev["type"] == "round":
            for m, d in ev["methods"].items():
                trajectories[m].append(d["accuracy"])
        elif ev["type"] == "final":
            metrics = ev["metrics"]
    return trajectories, metrics
