# Hospital Federated Learning · Security Lab

Live simulator: pick hospitals, malicious hospitals, defenses and attacks, then
watch the federated rounds happen in a 3D network view with live analysis.

## Run (3 steps)
```bash
pip install -r requirements.txt
python main.py
```
Then open **http://127.0.0.1:8000** in your browser (the backend serves the page itself).
The page also works if opened with VS Code Live Server or by double-clicking `index.html`: it finds
the backend on its own. The top-right pill shows what it found:
- green = connected
- amber = an OLD backend is running (replace all files from the zip, Ctrl+C, run `python main.py` again)
- red = no backend found (run `python main.py` in this folder)
The internet is needed once so the browser can load the Plotly charting library.

## Two computers: one attacker, one monitor
This is the live, two-person mode. Computer 1 gives the attacks, computer 2 watches and defends.

1. On the **monitor computer** run `python main.py`. It prints two addresses.
2. **Monitor**: open http://127.0.0.1:8000, choose **Mode: Live · 2 computers**, pick the dataset, hospitals,
   rounds and defenses, then press **Open live session**. The address for the attacker is shown on the left.
3. **Attacker computer** (same Wi-Fi or network): open the "Attacker screen" address that `main.py` printed,
   for example `http://192.168.1.20:8000/attacker`. The page joins the session by itself.
4. **Monitor** presses **Begin rounds**. The attacker picks an attack, picks hospitals (or 🎲 random ones) and
   presses **Launch**. A new attack counts from the next round that has not started yet, and it can be
   changed or stopped at any time.
5. The monitor sees, live: which hospitals are attacked and with which attack (red hospitals, the
   "under attack" banner, the roster and the **Attack feed**), what the detector flagged, each hospital's
   influence weight, and the best model of each round being sent back to the hospitals.
   When the session ends, the **Results** tab shows the best model delivered to every hospital (what each
   hospital reaches alone versus with the model it received) and a download button for the model.

Notes: this is a demo. There is no password, so use a trusted network. Allow Python through the firewall if
Windows asks. There is one live session at a time (opening a new one stops the old one). Finished live
sessions are saved in History and can be replayed.

## Datasets
Two real clinical datasets, chosen from the dataset box on the left: Pima Indians Diabetes and
Breast Cancer Wisconsin (Diagnostic). See `DATASETS.md` for details, literature precedent and results.

## Screen layout
- **Left**: sliders for hospitals (nodes), malicious hospitals, rounds, playback speed;
  defense and attack selection. The node preview updates as you move the sliders.
- **Centre**: live 3D hospital network, a 2D **network topology** (server + every hospital + links),
  accuracy round by round (3D ribbons or 2D lines), and a hospital x round detection map.
  The malicious hospitals are picked at random and a new set is drawn after every run (never side by side when there is room); press 🎲 Shuffle to re-draw.
- **Right**: live analysis (ranking, caught/false-alarm counts, hospital roster with
  influence weights), final results + plain-language insights, and run history.

## Three tiers
| Tier | Where |
|---|---|
| Front end | `index.html` (Plotly) |
| Back end | `main.py` (FastAPI, streams one JSON line per round) + `simulation.py` |
| Database | `database.py` (SQLite file `fl_runs.db`, created automatically) |

API: `POST /api/simulate/stream` (live), `POST /api/simulate` (one shot),
`GET /api/runs`, `GET /api/runs/{id}`, `DELETE /api/runs/{id}`.

Check the logic without a browser: `python test_run.py`
