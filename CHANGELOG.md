# Round 7: live two-computer mode (attacker + monitor) and best-model delivery

- New **attacker screen** (`/attacker`, `attacker.html`) for a second computer: choose one of the 10 attacks,
  choose hospitals (click, select all, or random), press Launch. It also shows which attacks are active
  and which ones the defense has noticed.
- New **Live · 2 computers** mode on the monitor screen: open a session, press Begin, and watch. Attacks
  given from the other computer take effect from the next round. New **Attack feed** tab, an
  "under attack now" banner (which hospital has which attack), and the roster/3D/topology show it live.
- Server listens on the local network (`0.0.0.0`) and prints the attacker address. New files: `live.py`
  (shared session) and endpoints under `/api/live/*`.
- **Best model delivery**: every round shows the best model so far "sent to all hospitals"; at the end a panel
  shows, per hospital, what it reaches alone versus with the delivered model, and the model can be downloaded.
  This also appears for normal single-screen runs.
- Engine: optional `attack_provider` (attacks decided live); rounds now carry the set of hospitals attacked in
  that round and the best model. Normal runs are unchanged (results verified identical, both datasets).
- Saved live sessions are replayable. Runs saved by older versions still replay.
- Tests: `python test_live.py` (live session with threads playing both computers).

# Round 6: malicious hospitals are random every run

- Which hospitals are malicious is now chosen at random, and a new set is drawn after every run
  (also with the new 🎲 Shuffle button, and whenever you move the node or malicious sliders).
  This works for any number of malicious hospitals. When there is room, no two malicious
  hospitals sit side by side (H1 and Hn count as neighbours on the ring). When that would leave
  fewer than 20 possible arrangements (e.g. 5 of 10 allows only 2), the hospitals are picked
  fully at random so the choice really changes.
- The page picks the set itself and sends it to the backend, so the dots and list on the left
  ("Next run: H2 · H5 · H9") are exactly what runs. If the backend is called without a set, it
  draws one at random. The same attackers are used for every defense inside one run.
- Each saved run stores which hospitals were malicious; History shows them.
- This replaces the fixed H1/H5/H10 placement of Round 4. Results are now different on every
  run, so report the mean and standard deviation over many runs (see DATASETS.md).

# Round 5: second dataset

- Added **Breast Cancer Wisconsin (Diagnostic)** (569 patients, 30 features, bundled with
  scikit-learn so nothing to download) next to Pima Indians Diabetes. Choose it in the Dataset box.
  Everything else (attacks, defenses, 3D views, topology, history) works on both.
- Backend: `DATASETS` registry and `load_dataset()` in `simulation.py`; `dataset` field in the API
  request; the choice is stored with every saved run. Pima results are unchanged (verified).
- Page: dataset selector; labels, results header and notes follow the dataset; replaying an old
  run switches back to that run's dataset (runs saved before this update count as Pima).
- New `DATASETS.md`: dataset details, papers that use them, how to cite, and the results
  of both datasets with identical, untuned settings.

# Round 4: network topology panel + spread-out malicious hospitals

- New **Network topology** panel under the 3D view: the central server in the centre, every
  hospital on a ring, a link to each one. Link thickness = influence on the global model,
  dotted = almost excluded, red = malicious (its current attack is written beside it),
  amber glow = flagged by the detector this round. Hover a hospital for details.
- Malicious hospitals are no longer neighbours. They are spread evenly across the network
  (10 hospitals, 3 malicious -> H1, H5, H10). The left-panel preview uses the same rule as the
  backend (checked for every hospital/malicious combination).
- Attacks are handed out by each malicious hospital's position in the list, so spread-out
  attackers still take turns with different attacks.
- Effect on results: with attackers all at one end of the glucose-sorted split (old placement),
  they hid among the extreme honest hospitals, the detector flagged 0% of them, and FedAvg
  collapsed to about 34-44%. With the spread placement the detector flags some attackers with
  no honest hospital wrongly flagged, FedAvg drops less (about 59-64%), and Soft Weighting
  reaches 71-74%. Numbers from earlier runs are not comparable with new runs.

# Round 3: live UI, 3D views and database

- New `index.html`: dark dashboard, left controls (node / malicious / rounds / speed
  sliders, defense + attack chips), centre 3D hospital network + 3D accuracy ribbons
  (2D toggle) + hospital x round detection heatmap, right analysis panel with
  live ranking, hospital roster (which hospital is malicious, which attack it is
  using this round, its z-score and influence weight), results, and history.
- Live streaming: `POST /api/simulate/stream` sends one JSON line per round
  (`simulation.run_simulation_stream`). `run_simulation` still works for tests.
- New `database.py`: every finished run is saved to SQLite (`fl_runs.db`) and can be
  replayed or deleted from the History tab. Standard library only.
- `aggregation/methods.py`: added `multi_krum_indices`, `strategy_a_kept`,
  `strategy_b_weights` so the UI can show each hospital's real influence weight.
  The formulas did not change; results were checked to be identical before/after.
- `main.py` serves the page at `/` and has history endpoints.

# Fixes and additions applied to this prototype

## Round 1 fixes (previous pass)
1. Real `diabetes.csv` loaded instead of fake synthetic `np.random.randn()` data.
2. Removed "FLDetector" -- it was byte-for-byte identical to "System A (Hard
   Exclusion)" (a copy-paste bug), confirmed by matching trajectories.
3. Classification metrics (diabetes-prediction quality) and Detection metrics
   (malicious-client-flagging quality) split into clearly separate, labeled
   dictionaries -- previously conflated under one `precision`/`recall`/`f1`.
4. `model_rescaling` renamed `gradient_scaling`; scale fixed to the spec'd
   value.

## Round 2: all 10 requested attacks, implemented to match their real
## published mechanisms (not placeholder weight-nudges)

| Attack | Mechanism | Citation / basis |
|---|---|---|
| `gradient_scaling` | delta = w_honest - w_global; w' = w_global + 10*delta | Model-poisoning / boosting (Bagdasaryan et al. 2020 family) |
| `sign_flipping` | w' = w_global - (w_honest - w_global) | Standard sign-flip, Byzantine literature |
| `free_rider` | w' = w_global + N(0, 0.5) -- no real training happens | Lin et al. free-rider; "disguised free-rider" variant |
| `gaussian_noise` | w' = w_honest + N(0, 1.0) | Generic Byzantine corruption |
| `byzantine` | w' = Uniform(-50, 50), fresh every round | True arbitrary/unstructured worst case (Blanchard et al. 2017, Yin et al. 2018 definition) |
| `sybil` | **Coordinated**: every hospital flagged "sybil" this round submits the SAME shared payload (built from their averaged honest gradients, boosted) | Fung et al. 2020 (FoolsGold) -- the defining trait of Sybil attacks is identical/highly-correlated updates from multiple identities |
| `label_flipping` | y' = 1 - y, applied before local training | Tolpegin et al. 2020 |
| `dirty_label_backdoor` | Real trigger inserted into a feature (Insulin forced to an extreme value) for ALL local samples, label forced to target class | Standard/dirty-label backdoor, Gu et al. / Bagdasaryan et al. 2020 |
| `clean_label_backdoor` | Same trigger, but ONLY inserted into samples that already belong to the target class -- labels stay truthful ("clean") | Clean-label backdoor variant, stealthier/less reliable per literature |
| `gradient_inversion_probe` | **Passive** -- hospital trains and submits honestly; a separate simplified gradient-matching reconstruction (mini-DLG, Zhu et al. 2019) scores how much of the hospital's data could be inferred from its gradient. Never affects model accuracy. | Zhu et al. 2019 (Deep Leakage from Gradients) |

### Update: Backdoor ASR metric removed
The Backdoor Attack Success Rate metric (column, API field and helper
function) was removed on request. The two backdoor attacks are still
selectable and still train on a real trigger, but the prototype no longer
measures trigger-specific behavior -- only overall accuracy, which a
backdoor is designed to leave unchanged.

### (Historical note) Attack Success Rate (ASR), now removed
For backdoor attacks, overall accuracy staying normal does NOT mean the
attack failed -- that's the whole point of a backdoor. `evaluate/metrics.py`
now builds a held-out triggered test set and reports ASR = the fraction of
genuinely non-target-class test patients that get flipped to the target
class once the trigger is present.

**Finding from testing this**: both backdoor variants achieve ~98% ASR,
completely unaffected by which defense is used (FedAvg, Hard Exclusion, and
Soft Weighting all show ~98% ASR). This is an honest, important limitation:
distance/MAD-based detection has no mechanism for catching backdoors, since
a backdoor only requires a small, localized weight change rather than a
large anomalous shift. This is consistent with the literature -- dedicated
backdoor defenses (e.g., FLAME) exist as a separate research line precisely
because generic Byzantine-robust aggregation does not catch them.

### Finding: Sybil collusion achieves 0% detection recall
Across all three methods tested, Sybil collusion is never flagged. This is
structural, not a tuning issue: when multiple hospitals submit an identical
update, they pull the detector's reference mean toward themselves rather
than appearing as individual outliers relative to it. This is exactly why
Sybil-specific defenses (FoolsGold) use **pairwise similarity between every
pair of clients** instead of distance-from-a-single-mean -- a fundamentally
different detection principle this prototype's detector does not implement.

## What's unchanged (already matched the core spec exactly)
- `federated/local_model.py` -- sigmoid, gradient, update formulas
- `detection/distance_detector.py` -- mean-based MAD z-score, tau=2.5
- `aggregation/methods.py` -- FedAvg, Multi-Krum, Strategy A, Strategy B

## Attacks documented but intentionally NOT implemented as live attacks
See `ATTACK_CATALOG.md` for the full reasoning -- Membership/Source
Inference attacks are passive privacy attacks that don't involve submitting
a poisoned update, so they don't fit this aggregation-robustness benchmark's
evaluation framework (they'd need a membership-inference evaluation
protocol instead). They're documented, not faked.
