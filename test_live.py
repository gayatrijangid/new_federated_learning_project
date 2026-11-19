"""Tests the live two-computer session with threads playing the roles of the
attacker computer and the monitor computer. Run:  python test_live.py"""
import json, threading, time, warnings, os
warnings.filterwarnings("ignore")
import database, live

passed = failed = 0
def check(name, cond):
    global passed, failed
    print(("PASS " if cond else "FAIL ") + name); passed += bool(cond); failed += (not cond)

def consume(session, sink, on_event=None):
    for line in session.stream():
        ev = json.loads(line)
        if ev["type"] == "ping": continue
        sink.append(ev)
        if on_event: on_event(ev)

METHODS = ["FedAvg", "Multi-Krum", "System B (Soft Weighting)"]
database.init_db()

# ======================= scenario 1: attacker + monitor + late joiner =======================
S = live.create_session(dataset="pima", rounds=8, num_hospitals=8, selected_methods=METHODS, delay_s=0.15)
monitor, attacker = [], []
action_log = {}

def attacker_brain(ev):
    if ev["type"] == "round" and ev["round"] == 3 and "h1" not in action_log:
        action_log["h1"] = S.set_attack(0, "label_flipping", "Attacker PC")
    if ev["type"] == "round" and ev["round"] == 5 and "clear" not in action_log:
        action_log["clear"] = S.set_attack(2, "none", "Attacker PC")

tm = threading.Thread(target=consume, args=(S, monitor)); ta = threading.Thread(target=consume, args=(S, attacker, attacker_brain))
tm.start(); ta.start()
time.sleep(0.2)
check("lobby: monitor and attacker both see the session before it begins", monitor and monitor[0]["type"] == "lobby" and attacker[0]["type"] == "lobby")
e1 = S.set_attack(2, "gradient_scaling", "Attacker PC"); e2 = S.set_attack(5, "sign_flipping", "Attacker PC")
check("attacks given in the lobby take effect from round 1", e1["effective_round"] == 1 and e2["effective_round"] == 1)
check("same attack again is a no-op (not logged twice)", S.set_attack(2, "gradient_scaling") is None)
S.begin()
tm.join(timeout=60); ta.join(timeout=60)

types = [e["type"] for e in monitor]
rounds = [e for e in monitor if e["type"] == "round"]
check("session finished", S.state()["status"] == "finished")
check("event order: lobby, attacks, start, 8 rounds, delivery, final, saved", types[0] == "lobby" and types.index("start") < types.index("delivery") < types.index("final") and len(rounds) == 8 and "saved" in types)
check("monitor and attacker received exactly the same events", monitor == attacker)
check("round 1 shows the lobby attacks: H3 gradient_scaling, H6 sign_flipping", rounds[0]["attacks"] == {"2": "gradient_scaling", "5": "sign_flipping"} and rounds[0]["malicious_ids"] == [2, 5])
eff = action_log["h1"]["effective_round"]
first_h1 = next(r["round"] for r in rounds if "0" in r["attacks"])
check(f"attack given during round 3 starts exactly at its announced round ({eff})", first_h1 == eff and eff > 3 and all("0" not in r["attacks"] for r in rounds if r["round"] < eff))
eff_c = action_log["clear"]["effective_round"]
check("clearing an attack stops it at the announced round", all("2" not in r["attacks"] for r in rounds if r["round"] >= eff_c) and all("2" in r["attacks"] for r in rounds if r["round"] < eff_c))
atk_events = [e for e in monitor if e["type"] == "attack"]
check("attack log names the hospital, the attack and who did it", [(e["hospital"], e["attack"], e["actor"]) for e in atk_events][:2] == [(2, "gradient_scaling", "Attacker PC"), (5, "sign_flipping", "Attacker PC")])
dl = next(e for e in monitor if e["type"] == "delivery")
att = {h["id"]: h["attacked"] for h in dl["hospitals"]}
check("delivery: best model goes to all 8 hospitals; attacked ones are marked", len(dl["hospitals"]) == 8 and att[0] and att[2] and att[5] and not att[1])
check("delivery carries the model weights (9 numbers: 8 features + bias)", len(dl["model"]["weights"]) == 9)
late = []; consume(S, late)
check("a screen that joins AFTER the end still gets the full history", late == monitor)
check("run was saved to the database with the live flag", database.get_run(S.run_id) is not None and database.list_runs()[0]["config"].get("live") is True)
raised = []
for bad in [(99, "sybil"), (1, "no_such_attack"), ("x", "sybil")]:
    try: S.set_attack(*bad)
    except ValueError as e: raised.append(str(e))
check("bad hospital / bad attack / non-number are rejected with a message", len(raised) == 3)
try: S.set_attack(1, "sybil"); ended = False
except ValueError: ended = True
check("cannot attack after the session has ended", ended)

# ======================= scenario 2: pause / resume / stop =======================
S2 = live.create_session(dataset="breast_cancer", rounds=60, num_hospitals=6, selected_methods=["FedAvg"], delay_s=0.1)
ev2 = []; t2 = threading.Thread(target=consume, args=(S2, ev2)); t2.start()
S2.begin(); time.sleep(0.8)
S2.control("pause"); time.sleep(0.4); r_at_pause = S2.state()["round"]; time.sleep(0.8)
check("pause freezes the rounds", S2.state()["round"] - r_at_pause <= 1 and S2.state()["status"] == "paused")
S2.control("resume"); time.sleep(0.8)
check("resume continues the rounds", S2.state()["round"] > r_at_pause + 1)
S2.control("stop"); t2.join(timeout=10)
check("stop ends the session and the screens' streams", S2.state()["status"] == "stopped" and not t2.is_alive())
check("creating a new session stops the old one", (lambda s3: live.get_session() is s3)(live.create_session(rounds=5)) and S2._done())
S4 = live.create_session(rounds=5); S4.control("stop")
check("stopping in the lobby works", S4.state()["status"] == "stopped")

print(f"\n{passed} passed, {failed} failed")
if os.path.exists(database.DB_PATH): os.remove(database.DB_PATH)
raise SystemExit(1 if failed else 0)
