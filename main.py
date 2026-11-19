import json
import os
import socket
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import database
import live
from simulation import (run_simulation, run_simulation_stream,
                        VALID_METHODS, VALID_ATTACKS, DATASETS, DEFAULT_DATASET)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Hospital Federated Learning Defense API")

# Allows the page to work whether you open http://127.0.0.1:8000 (recommended)
# or double-click index.html directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()


class SimConfig(BaseModel):
    dataset: str = DEFAULT_DATASET
    malicious_ids: Optional[List[int]] = None   # None = pick a fresh random set
    rounds: int = 30
    num_hospitals: int = 10
    num_malicious: int = 3
    selected_methods: List[str] = VALID_METHODS
    attack_types: List[str] = ["sign_flipping", "label_flipping"]


def lan_ips():
    """This computer's address(es) on the local network, so the attacker's computer knows where to connect."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                    # nothing is sent; this just finds the active network card
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


@app.get("/")
def home():
    """Serves the UI, so just open http://127.0.0.1:8000 in the browser."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/attacker")
def attacker_page():
    """The attacker's screen (open this one on the OTHER computer)."""
    return FileResponse(os.path.join(BASE_DIR, "attacker.html"))


@app.get("/api/network")
def network_info(request: Request):
    port = request.url.port or 8000
    return {"port": port, "urls": ["http://%s:%s/attacker" % (ip, port) for ip in lan_ips()]}


@app.get("/api/options")
def get_options():
    return {"methods": VALID_METHODS, "attacks": VALID_ATTACKS, "datasets": DATASETS, "live": True}


@app.post("/api/simulate")
def simulate(config: SimConfig):
    """Classic one-shot endpoint: runs everything, returns the final result."""
    trajectories, metrics = run_simulation(
        rounds=config.rounds,
        num_hospitals=config.num_hospitals,
        num_malicious=config.num_malicious,
        selected_methods=config.selected_methods,
        attack_types=config.attack_types,
        dataset=config.dataset,
        malicious_ids=config.malicious_ids,
    )
    return {"status": "success", "trajectories": trajectories, "metrics": metrics}


@app.post("/api/simulate/stream")
def simulate_stream(config: SimConfig):
    """Live endpoint: sends one JSON line per round while the simulation runs
    (newline-delimited JSON). The UI draws each round as it arrives. When the
    run finishes it is saved to the SQLite database (History tab)."""
    cfg_dict = {
        "dataset": config.dataset,
        "rounds": config.rounds,
        "num_hospitals": config.num_hospitals,
        "num_malicious": config.num_malicious,
        "selected_methods": config.selected_methods,
        "attack_types": config.attack_types,
    }

    def event_stream():
        events = []
        try:
            for ev in run_simulation_stream(
                rounds=config.rounds,
                num_hospitals=config.num_hospitals,
                num_malicious=config.num_malicious,
                selected_methods=config.selected_methods,
                attack_types=config.attack_types,
                dataset=config.dataset,
                malicious_ids=config.malicious_ids,
            ):
                events.append(ev)
                out = ev
                if ev["type"] == "final":
                    run_id = database.save_run(events[0]["config"], events, ev["metrics"])
                    out = {**ev, "run_id": run_id}
                yield json.dumps(out) + "\n"
        except Exception as exc:  # report the problem to the UI instead of dying silently
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ===================== LIVE SESSION: attacker computer + monitor computer =====================
class LiveOpenConfig(BaseModel):
    dataset: str = DEFAULT_DATASET
    rounds: int = 30
    num_hospitals: int = 10
    selected_methods: List[str] = VALID_METHODS
    delay_s: float = 2.0


class AttackItem(BaseModel):
    hospital: int
    attack: Optional[str] = None      # None / "none" removes the attack


class AttackCommand(BaseModel):
    assignments: List[AttackItem]
    actor: Optional[str] = "attacker"


class ControlCommand(BaseModel):
    action: str                        # begin | pause | resume | stop | delay
    delay_s: Optional[float] = None


def _session_or_404():
    s = live.get_session()
    if s is None:
        raise HTTPException(status_code=404, detail="No live session is open yet.")
    return s


@app.post("/api/live/open")
def live_open(config: LiveOpenConfig):
    """The monitor opens a session (lobby). Rounds start when it presses Begin."""
    s = live.create_session(dataset=config.dataset, rounds=config.rounds, num_hospitals=config.num_hospitals,
                            selected_methods=config.selected_methods, delay_s=config.delay_s)
    return s.state()


@app.get("/api/live/state")
def live_state():
    s = live.get_session()
    return s.state() if s else {"exists": False}


@app.get("/api/live/stream")
def live_stream():
    """Every screen connects here: it receives the whole history, then live events."""
    s = _session_or_404()
    return StreamingResponse(s.stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/live/attack")
def live_attack(cmd: AttackCommand):
    """The attacker gives attacks to hospitals (takes effect from the next round)."""
    s = _session_or_404()
    changed = 0
    try:
        for item in cmd.assignments:
            if s.set_attack(item.hospital, item.attack, cmd.actor or "attacker") is not None:
                changed += 1
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"changed": changed, "state": s.state()}


@app.post("/api/live/control")
def live_control(cmd: ControlCommand):
    s = _session_or_404()
    if cmd.action not in ("begin", "pause", "resume", "stop", "delay"):
        raise HTTPException(status_code=400, detail="Unknown action.")
    s.control(cmd.action, cmd.delay_s)
    return s.state()


@app.get("/api/runs")
def list_runs():
    return {"runs": database.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    run = database.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: int):
    if not database.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    print("\n  Monitor screen (this computer):          http://127.0.0.1:8000")
    for ip in lan_ips():
        print("  Attacker screen (the OTHER computer):    http://%s:8000/attacker" % ip)
    print("  (both computers must be on the same Wi-Fi / network)\n")
    # 0.0.0.0 = reachable from other computers on the same network
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
