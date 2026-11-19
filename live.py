"""
live.py
========
A LIVE session shared by two (or more) computers.

  * Computer 2 (the monitor / defender) opens a session, then presses Begin.
  * Computer 1 (the attacker) joins the same session and gives attacks to
    hospitals while the rounds are running.
  * Every round the engine reads the CURRENT attacks, so a new attack takes
    effect from the next round. All screens see the same events, live.

Everything a screen needs is sent as one JSON line per event (see stream()).
A screen that joins late first receives the whole history, then live events.
"""
import json
import threading
import time

import database
from simulation import (run_simulation_stream, VALID_ATTACKS, VALID_METHODS,
                        DATASETS, DEFAULT_DATASET)


class LiveSession:
    def __init__(self, dataset=DEFAULT_DATASET, rounds=30, num_hospitals=10,
                 selected_methods=None, delay_s=1.5):
        self.dataset = dataset if dataset in DATASETS else DEFAULT_DATASET
        self.rounds = int(max(1, min(200, rounds)))
        self.num_hospitals = int(max(3, min(40, num_hospitals)))
        self.methods = [m for m in (selected_methods or VALID_METHODS) if m in VALID_METHODS] or list(VALID_METHODS)
        self.delay_s = float(max(0.0, min(15.0, delay_s)))

        self.id = int(time.time() * 1000)   # lets a screen notice that a NEW session was opened
        self.cond = threading.Condition()
        self.events = []
        self.assignments = {}          # hospital id -> attack name (the attacks happening NOW)
        self.status = "lobby"          # lobby | running | paused | finished | stopped | error
        self.round = 0                 # rounds finished so far
        self.snapshotted = 0           # last round whose attacks have already been read by the engine
        self.run_id = None
        self._stop = False
        self._paused = False
        self._thread = None
        self._publish({
            "type": "lobby",
            "config": {"dataset": self.dataset, "rounds": self.rounds, "num_hospitals": self.num_hospitals,
                       "selected_methods": self.methods, "delay_s": self.delay_s, "live": True},
        })

    # ------------------------------------------------------------------ events
    def _publish(self, ev):
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def _set_status(self, status):
        with self.cond:
            self.status = status
        self._publish({"type": "session", "status": status, "round": self.round})

    def _done(self):
        return self.status in ("finished", "stopped", "error")

    def stream(self):
        """Yields JSON lines: the full history first, then live events, until the session ends.
        A 'ping' line is sent every 10 s of silence so connections stay open."""
        i = 0
        while True:
            with self.cond:
                while i >= len(self.events) and not self._done():
                    if not self.cond.wait(timeout=10):
                        break
                chunk = self.events[i:]
                i += len(chunk)
                finished = self._done() and i >= len(self.events)
            if not chunk and not finished:
                yield json.dumps({"type": "ping"}) + "\n"
            for ev in chunk:
                yield json.dumps(ev) + "\n"
            if finished:
                return

    # ------------------------------------------------------------------ attacks
    def _provider(self, r):
        """Called by the engine at the start of each round: the attacks happening now."""
        with self.cond:
            self.snapshotted = r
            return dict(self.assignments)

    def set_attack(self, hospital, attack, actor="attacker"):
        """Give an attack to a hospital (attack=None/'none' removes it). Takes effect from the
        next round that has not started yet. Returns the logged event (or None if nothing changed)."""
        if self._done():
            raise ValueError("The session has already ended.")
        try:
            hospital = int(hospital)
        except (TypeError, ValueError):
            raise ValueError("Hospital must be a number.")
        if not 0 <= hospital < self.num_hospitals:
            raise ValueError(f"Hospital must be between 0 and {self.num_hospitals - 1}.")
        attack = None if attack in (None, "", "none") else attack
        if attack is not None and attack not in VALID_ATTACKS:
            raise ValueError(f"Unknown attack '{attack}'.")
        with self.cond:
            if self.assignments.get(hospital) == attack or (attack is None and hospital not in self.assignments):
                return None
            if attack is None:
                self.assignments.pop(hospital, None)
            else:
                self.assignments[hospital] = attack
            ev = {"type": "attack", "hospital": hospital, "attack": attack,
                  "effective_round": self.snapshotted + 1, "after_round": self.round,
                  "actor": str(actor)[:40] or "attacker"}
            self.events.append(ev)
            self.cond.notify_all()
        return ev

    # ------------------------------------------------------------------ control
    def begin(self):
        with self.cond:
            if self.status != "lobby":
                return False
            self.status = "running"
        self._publish({"type": "session", "status": "running", "round": 0})
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def control(self, action, delay_s=None):
        if delay_s is not None:
            self.delay_s = float(max(0.0, min(15.0, delay_s)))
        if action == "begin":
            return self.begin()
        if action == "pause" and self.status == "running":
            self._paused = True
            self._set_status("paused")
        elif action == "resume" and self.status == "paused":
            self._paused = False
            self._set_status("running")
        elif action == "stop" and not self._done():
            self._stop = True
            self._paused = False
            if self._thread is None:                 # still in the lobby: nothing is running
                self._set_status("stopped")
        return True

    def state(self):
        with self.cond:
            return {
                "exists": True, "id": self.id, "status": self.status, "round": self.round, "rounds": self.rounds,
                "next_round": self.snapshotted + 1,
                "num_hospitals": self.num_hospitals, "dataset": self.dataset, "methods": self.methods,
                "delay_s": self.delay_s, "run_id": self.run_id,
                "assignments": {str(h): a for h, a in self.assignments.items()},
            }

    # ------------------------------------------------------------------ the engine thread
    def _run(self):
        try:
            gen = run_simulation_stream(self.rounds, self.num_hospitals, 0, self.methods, [],
                                        self.dataset, malicious_ids=[], attack_provider=self._provider)
            final_metrics = None
            for ev in gen:
                if self._stop:
                    break
                if ev["type"] == "final":
                    final_metrics = ev["metrics"]
                if ev["type"] == "round":
                    self.round = ev["round"]
                self._publish(ev)
                if ev["type"] == "round":              # give the humans time to react
                    t0 = time.time()
                    while time.time() - t0 < self.delay_s and not self._stop:
                        time.sleep(0.03)
                    while self._paused and not self._stop:
                        time.sleep(0.1)
            if self._stop:
                self._set_status("stopped")
                return
            if final_metrics is not None:
                try:
                    start_cfg = dict(self.events[[e["type"] for e in self.events].index("start")]["config"])
                    with self.cond:
                        snapshot = list(self.events)
                    self.run_id = database.save_run(start_cfg, snapshot, final_metrics)
                    self._publish({"type": "saved", "run_id": self.run_id})
                except Exception:                       # saving history must never break the session
                    pass
            self._set_status("finished")
        except Exception as exc:
            self._publish({"type": "error", "message": str(exc)})
            self._set_status("error")


# ---------------------------------------------------------------------- one shared session
_session = None
_lock = threading.Lock()


def create_session(**kwargs):
    """Opens a new session (any previous one is stopped)."""
    global _session
    with _lock:
        if _session is not None and not _session._done():
            _session.control("stop")
        _session = LiveSession(**kwargs)
        return _session


def get_session():
    with _lock:
        return _session
