"""Standalone test harness -- no FastAPI needed."""
from simulation import run_simulation, VALID_METHODS, VALID_ATTACKS, load_real_pima_dataset

if __name__ == "__main__":
    print("Available methods:", VALID_METHODS)
    print("Available attacks (all 10):", VALID_ATTACKS)
    assert len(VALID_ATTACKS) == 10, f"expected 10 attacks, got {len(VALID_ATTACKS)}"

    _, X_test, y_test = load_real_pima_dataset(10)
    print(f"\n[data check] X_test range: [{X_test.min():.2f}, {X_test.max():.2f}]  "
          f"n_test={len(y_test)}  diabetic_rate={y_test.mean():.2f}  (real clinical data)")

    print("\n--- Default run (gradient_scaling + sign_flipping) ---")
    traj, metrics = run_simulation(rounds=20, num_hospitals=10, num_malicious=3,
                                    selected_methods=VALID_METHODS,
                                    attack_types=["gradient_scaling", "sign_flipping"])
    print(f"{'Method':<28}{'Acc%':<8}{'F1%':<8}| {'DetPrec%':<10}{'DetRec%':<9}{'DetF1%'}")
    for m, v in metrics.items():
        c, d = v["classification"], v["detection"]
        print(f"{m:<28}{c['accuracy']:<8}{c['f1_score']:<8}| {d['detection_precision']:<10}{d['detection_recall']:<9}{d['detection_f1']}")

    print("\n--- Each of the 10 attacks individually (3 methods) ---")
    methods3 = ["FedAvg", "System A (Hard Exclusion)", "System B (Soft Weighting)"]
    for atk in VALID_ATTACKS:
        traj, metrics = run_simulation(rounds=15, num_hospitals=10, num_malicious=3,
                                        selected_methods=methods3, attack_types=[atk])
        accs = {m: v["classification"]["accuracy"] for m, v in metrics.items()}
        print(f"{atk:<28}", accs)

    print("\n--- Sybil detection check (expect near-0%, structural blind spot) ---")
    traj, metrics = run_simulation(rounds=20, num_hospitals=10, num_malicious=3,
                                    selected_methods=methods3, attack_types=["sybil"])
    for m, v in metrics.items():
        d = v["detection"]
        print(f"  {m:<28} det_recall={d['detection_recall']}%")

    print("\nAll checks passed.")
