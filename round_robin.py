"""Cross-branch tournament coordinator over a shared pile of agent bundles.

benchmark.py compares checkpoints that share this checkout's code. This
coordinator compares *agents that need not share any code at all*: each
contestant is an exported bundle (export_agent.py) run as a black-box
subprocess speaking the agent_server.py stdio protocol. The coordinator owns
the one battle engine both sides play on (its own sidecar, its own format
pin — fair by construction), assigns the replica teams with the same pairing
grid and side alternation as benchmark.py, and shuttles protocol lines and
requests between the sim and the two agent processes. What is inside an agent
— transformer, hand-coded search, anything — is invisible, so branches with
incompatible tokenizers or action spaces can still play each other.

Results append to <pile>/results.jsonl, so they travel with the pile when it
is rsync'd between machines. Fairness across architectures is reported, not
assumed: every row records each side's average seconds per move; pass
--move-budget S to also *enforce* a per-move wall clock (a late agent plays
Showdown's "default" for that request). A hung or crashed agent forfeits the
game and keeps a stderr log under <pile>/logs/.

Schedules: `play A B` runs one pairing over the full replica-team grid;
`star` runs every bundle against an anchor (default: the bundle named
"baseline") — this keeps the Bradley-Terry graph connected at N series
instead of N(N-1)/2; `all` is the full round robin, best saved for finalists.

play/star/all default to a live web dashboard (spectate.py, port 8020 — the
opened port triggers VS Code's forward popup): per-game trackers you can flip
between, plus a control panel — skip the current matchup (e.g. random vs
baseline dragging on), grow/shrink the worker pool between games, pause and
resume, live pairing score/ETA and pile standings. `--no-live` restores the
headless run.

CLI: python round_robin.py list [--pile P]
     python round_robin.py play A B [options]
     python round_robin.py star [--anchor NAME] [options]
     python round_robin.py all [options]
     python round_robin.py standings [--pile P]
Options: --pile P --workers N --quick N --repeat N --move-budget S
         --hang-timeout S --temp T --label TAG --no-save
         --no-live --dash-port P
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("round_robin.py"):
        raise SystemExit(0)

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

import teams as teams_mod
from benchmark import elo_diff, git_commit, series_pairings, wilson
from config import CFG
from env import Sidecar, SidecarBattle, pack_team
from observe_game import cts_placeholder

PROTOCOL_VERSION = 1
HELLO_TIMEOUT_S = 600.0    # covers torch import + model load + node spawn
GAME_READY_TIMEOUT_S = 120.0
BUDGET_GRACE_S = 5.0       # network/queue slack on an enforced move budget


def pile_dir(explicit=None):
    """Resolve the pile directory: flag, then $VGC_PILE, then ../vgc-pile."""
    root = Path(__file__).resolve().parent
    return Path(explicit or os.environ.get("VGC_PILE") or root.parent / "vgc-pile")


def load_pile(pile):
    """Return {name: {"dir": Path, "manifest": dict}} for every bundle."""
    out = {}
    p = Path(pile)
    if not p.exists():
        return out
    for d in sorted(p.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            out[d.name] = {"dir": d, "manifest": json.loads(m.read_text())}
    return out


def ledger_path(pile):
    """Return the append-only results file inside the pile."""
    return Path(pile) / "results.jsonl"


def read_ledger(pile):
    """Return every recorded result row (empty list when none)."""
    p = ledger_path(pile)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line]


class AgentDied(Exception):
    """A contestant process crashed or hung; the game is a forfeit."""


class TournamentControls:
    """Runner-owned live control state behind the spectator dashboard.

    The dashboard (spectate.py) calls ``state()`` for its panel and
    ``action()`` for its buttons; ``run_pairing``'s worker pool obeys the
    resulting flags: *skip matchup* drains the current pairing's queued jobs
    (in-flight games finish and are still recorded), *workers +/-* grows or
    shrinks the pool between games (each worker owns a sidecar plus one
    subprocess per contestant, so growth is capped), *pause* stops workers
    from picking up new games without killing anything."""

    MAX_WORKERS = 8

    def __init__(self, workers, pile=None, budget=None):
        """Set initial worker target and the pile used for live standings."""
        self.lock = threading.Lock()
        self.target_workers = max(1, int(workers))
        self.paused = False
        self.pile, self.budget = pile, budget
        self.plan_done = 0
        self.plan_total = 1
        self._live = None                  # attached by run_pairing
        self._standings = (0.0, [])        # (fetched_at, rows) 10s cache

    def attach(self, label, jobs, results, lock, t0, alive):
        """Register the running pairing's shared structures; return None."""
        with self.lock:
            self._live = {"label": label, "jobs": jobs, "results": results,
                          "lock": lock, "t0": t0, "alive": alive,
                          "total": len(jobs)}

    def detach(self):
        """Mark the current pairing finished; return None."""
        with self.lock:
            self._live = None
            self.plan_done += 1

    def state(self):
        """Return the JSON-safe control-panel snapshot."""
        with self.lock:
            live = self._live
            out = {"paused": self.paused, "workers_target": self.target_workers,
                   "budget_s": self.budget, "plan_done": self.plan_done,
                   "plan_total": self.plan_total, "pairing": None,
                   "workers_alive": 0, "standings": self._standings_rows()}
        if live:
            with live["lock"]:
                done = len(live["results"])
                out.update(
                    pairing=live["label"], done=done, total=live["total"],
                    jobs_left=len(live["jobs"]),
                    workers_alive=len(live["alive"]),
                    wins_a=sum(r["winner"] == "a" for r in live["results"]),
                    wins_b=sum(r["winner"] == "b" for r in live["results"]),
                    ties=sum(r["winner"] == "tie" for r in live["results"]))
            if done:
                per = (time.time() - live["t0"]) / done
                out["s_per_game"] = per
                out["eta_min"] = round(per * out["jobs_left"] / 60)
        return out

    def action(self, cmd, arg=""):
        """Apply one dashboard action; return a human-readable message."""
        if cmd == "skip":
            with self.lock:
                live = self._live
            if not live:
                return "nothing to skip"
            with live["lock"]:
                n = len(live["jobs"])
                live["jobs"].clear()
            return f"skipped {n} queued games of {live['label']} " \
                   "(in-flight games finish)"
        if cmd == "workers":
            with self.lock:
                self.target_workers = min(
                    self.MAX_WORKERS,
                    max(1, self.target_workers + int(arg or 0)))
                return f"worker target -> {self.target_workers}"
        if cmd in ("pause", "resume"):
            with self.lock:
                self.paused = cmd == "pause"
            return f"{cmd}d"
        return f"unknown control '{cmd}'"

    def _standings_rows(self):
        """Return dashboard standings rows, refreshed at most every 10 s.
        Callers hold self.lock."""
        now = time.time()
        if self.pile and now - self._standings[0] > 10:
            try:
                self._standings = (now, bt_rows(read_ledger(self.pile))[:12])
            except Exception:
                self._standings = (now, [])
        return self._standings[1]


class AgentProcess:
    """One live agent bundle as a subprocess speaking protocol v1."""

    def __init__(self, name, bundle, log_path, hang_timeout):
        """Spawn the manifest entrypoint with a shared Node dir and log."""
        man = bundle["manifest"]
        assert man.get("protocol", 1) == PROTOCOL_VERSION, \
            f"{name}: bundle speaks protocol {man.get('protocol')}, " \
            f"coordinator speaks {PROTOCOL_VERSION}"
        cmd = list(man["entrypoint"])
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable          # shared-venv policy
        env = dict(os.environ)
        env["VGC_NODE_DIR"] = str(Path(CFG.node_dir).resolve())
        env["PYTHONUNBUFFERED"] = "1"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._errlog = open(log_path, "a", encoding="utf-8")
        self.name, self.log_path = name, log_path
        self.hang_timeout = hang_timeout
        self.proc = subprocess.Popen(
            cmd, cwd=str(bundle["dir"] / "src"), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errlog, text=True, encoding="utf-8")
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.send({"type": "hello", "protocol": PROTOCOL_VERSION})
        ready = self.recv(HELLO_TIMEOUT_S)
        if not ready or ready.get("type") != "ready":
            raise AgentDied(f"{name}: no ready handshake (see {log_path})")

    def _reader(self):
        """Queue each stdout JSON line; a sentinel marks EOF (process died)."""
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except ValueError:
                pass                        # non-JSON garbage: ignore
        self.q.put({"type": "_eof"})

    def send(self, msg):
        """Write one protocol message; raises AgentDied on a closed pipe."""
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AgentDied(f"{self.name}: pipe closed ({exc})") from exc

    def recv(self, timeout):
        """Return the next message within ``timeout`` seconds, else None."""
        try:
            msg = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if msg.get("type") == "_eof":
            raise AgentDied(f"{self.name}: process exited "
                            f"(code {self.proc.poll()}, see {self.log_path})")
        return msg

    def ask(self, rqid, request, budget):
        """Forward one sim request; return (choice, wall_s, error, late).

        With a budget, a reply later than budget+grace plays "default" and is
        marked late (the stale reply is discarded by rqid when it arrives).
        Without one, silence past hang_timeout raises AgentDied."""
        self.send({"type": "request", "rqid": rqid, "request": request,
                   "deadline_s": budget})
        wait = (budget + BUDGET_GRACE_S) if budget else self.hang_timeout
        deadline = time.monotonic() + wait
        while True:
            msg = self.recv(max(0.0, deadline - time.monotonic()))
            if msg is None:
                if budget:
                    return "default", wait, None, True
                raise AgentDied(f"{self.name}: no reply in {wait:.0f}s")
            if msg.get("type") == "choice" and msg.get("rqid") == rqid:
                return (msg.get("choice", "default"),
                        float(msg.get("wall_s", 0.0)),
                        msg.get("error"), False)
            # stale rqid or unexpected type: drop and keep waiting

    def start_game(self, side, team, opp_preview, seed, temperature):
        """Send game_start and wait for the game_ready acknowledgement."""
        self.send({"type": "game_start", "side": side, "team": team,
                   "opp_preview": opp_preview, "seed": seed,
                   "temperature": temperature})
        msg = self.recv(GAME_READY_TIMEOUT_S)
        if not msg or msg.get("type") != "game_ready":
            raise AgentDied(f"{self.name}: no game_ready acknowledgement")

    def lines(self, raw_lines):
        """Stream protocol lines to the agent (fire and forget)."""
        self.send({"type": "lines", "lines": list(raw_lines)})

    def game_end(self, winner):
        """Notify the agent that the game ended (best effort)."""
        try:
            self.send({"type": "game_end", "winner": winner})
        except AgentDied:
            pass

    def close(self):
        """Best-effort quit, then kill; close the stderr log."""
        try:
            self.send({"type": "quit"})
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self._errlog.close()


def play_rpc_game(sc, agents, sets_by_side, cfg, seed, temperature, budget,
                  feed=None, max_turns=300):
    """One coordinated game between two agent processes.

    agents: {"p1": AgentProcess, "p2": AgentProcess}. Returns
    (winner side id or None, turns, per-side stats, forfeit side or None)."""
    b = SidecarBattle.create(sc, cfg.format_id,
                             pack_team(sets_by_side["p1"]),
                             pack_team(sets_by_side["p2"]))
    stats = {s: {"moves": 0, "wall": 0.0, "errors": 0, "late": 0,
                 "illegal": 0} for s in ("p1", "p2")}
    winner, forfeit, turns = None, None, 0

    def broadcast(raw_lines):
        """Send one log chunk to both agents and the optional spectator."""
        if feed:
            feed.feed(raw_lines)
        for ap in agents.values():
            ap.lines(raw_lines)

    try:
        for side, ap in agents.items():
            opp = "p2" if side == "p1" else "p1"
            ap.start_game(side, sets_by_side[side],
                          [cts_placeholder(s) for s in sets_by_side[opp]],
                          seed, temperature)
        broadcast(b.log)
        rqid = 0
        while not b.ended:
            choices = {}
            for side in b.pending_sides():
                rqid += 1
                choice, wall, error, late = agents[side].ask(
                    rqid, b.requests[side], budget)
                st = stats[side]
                st["moves"] += 1
                st["wall"] += wall
                st["errors"] += bool(error)
                st["late"] += late
                choices[side] = choice
            resp = b.step(choices)
            broadcast(resp["log"])
            if resp["errors"]:            # illegal choice: sim-default fallback
                for side in resp["errors"]:
                    stats[side]["illegal"] += 1
                resp = b.step({s: "default" for s in resp["errors"]})
                broadcast(resp["log"])
            turns += 1
            if turns >= max_turns:        # stall war: tie, don't hang
                break
        winner = b.winner if b.ended else None
    except AgentDied as exc:
        # attribute the forfeit to whichever side's process failed
        forfeit = next((s for s, ap in agents.items()
                        if str(exc).startswith(ap.name)), None)
        if forfeit:
            winner = "p2" if forfeit == "p1" else "p1"
        print(f"  forfeit: {exc}")
    finally:
        b.destroy()
    for ap in agents.values():
        ap.game_end(winner)
    if feed:
        feed.finish(winner)
    return winner, turns, stats, forfeit


def run_pairing(name_a, name_b, pile, bundles, cfg=CFG, workers=2, quick=None,
                repeat=1, budget=None, hang_timeout=300.0, temperature=0.0,
                label=None, save_replays=True, verbose=True, spectator=None,
                controls=None):
    """One full series between two bundles; append rows to the pile ledger.

    Mirrors benchmark.run_series: every ordered replica-team pairing, engine
    sides alternating by game parity, per-game seeds derived from the game
    index so a series is repeatable up to search nondeterminism.

    ``spectator``/``controls`` are normally created once by ``main`` and
    shared across a star/all run so one dashboard covers every pairing;
    without them behavior is the old headless save-only run. With controls
    the worker pool is dynamic: it obeys the live worker target, the pause
    flag, and the skip action (which drains this pairing's queued jobs)."""
    assert name_a != name_b, \
        "mirror matches need two exports of the same bundle under two names " \
        "(one process cannot play both sides of a game)"
    team_names = list(teams_mod.TEAMS)
    team_sets = {t: teams_mod.get(t) for t in team_names}
    jobs = [(g, ta, tb) for g, (ta, tb)
            in enumerate(series_pairings(team_names, repeat, quick))]
    run_tag = f"{name_a}_vs_{name_b}" + (f"_{label}" if label else "")
    run_id = f"{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if spectator is None and save_replays:
        try:
            from spectate import Spectator
            spectator = Spectator(run_tag, cfg, live=False, save=True)
            print(f"  saving replays under {spectator.dir}/")
        except Exception as exc:            # replays are never load-bearing
            print(f"  replay saving unavailable: {exc}")

    lock, results, t0 = threading.Lock(), [], time.time()
    alive = set()                           # live worker ids, guarded by lock
    coordinator_commit = git_commit()
    if controls:
        controls.attach(f"{name_a} vs {name_b}", jobs, results, lock, t0,
                        alive)

    def row_for(g, ta, tb, side_of, winner, turns, stats, forfeit):
        """Build one ledger row from a finished game."""
        of = {v: k for k, v in side_of.items()}       # side id -> "a"/"b"
        row = {"ts": datetime.now().isoformat(timespec="seconds"),
               "run": run_id, "a": name_a, "b": name_b,
               "team_a": ta, "team_b": tb,
               "winner": of.get(winner, "tie"),
               "forfeit": of.get(forfeit), "turns": turns,
               "seed": (g << 8) + 1, "format": cfg.format_id,
               "budget_s": budget, "temp": temperature,
               "coordinator_commit": coordinator_commit,
               "date": date.today().isoformat()}
        for tag, name in (("a", name_a), ("b", name_b)):
            man = bundles[name]["manifest"]
            st = stats[side_of[tag]]
            row[f"arch_{tag}"] = man.get("architecture", "?")
            row[f"commit_{tag}"] = man.get("git", {}).get("commit", "")
            row[f"moves_{tag}"] = st["moves"]
            row[f"wall_{tag}"] = round(st["wall"], 3)
            row[f"errors_{tag}"] = st["errors"]
            row[f"late_{tag}"] = st["late"]
            row[f"illegal_{tag}"] = st["illegal"]
        return row

    def worker(wid):
        """Own one sidecar + one process per contestant; drain the job list."""
        sc = Sidecar(cfg)
        procs = {}
        logs = Path(pile) / "logs" / run_id

        def proc_for(name):
            """Return this worker's live process for a bundle (spawn once)."""
            if name not in procs:
                procs[name] = AgentProcess(
                    name, bundles[name], logs / f"{name}.w{wid}.log",
                    hang_timeout)
            return procs[name]

        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    if controls and len(alive) > controls.target_workers:
                        alive.discard(wid)      # pool shrunk: retire quietly
                        return
                    job = None if controls and controls.paused \
                        else jobs.pop(0)
                if job is None:
                    time.sleep(0.5)             # paused: hold, don't exit
                    continue
                g, ta, tb = job
                side_of = {"a": "p1", "b": "p2"} if g % 2 == 0 else \
                          {"a": "p2", "b": "p1"}
                sets = {side_of["a"]: team_sets[ta],
                        side_of["b"]: team_sets[tb]}
                agents = {side_of["a"]: proc_for(name_a),
                          side_of["b"]: proc_for(name_b)}
                fd = spectator.new_game(name_a, name_b, ta, tb, side_of,
                                        cfg.format_id) if spectator else None
                winner, turns, stats, forfeit = play_rpc_game(
                    sc, agents, sets, cfg, seed=(g << 8) + 1,
                    temperature=temperature, budget=budget, feed=fd)
                row = row_for(g, ta, tb, side_of, winner, turns, stats,
                              forfeit)
                with lock:
                    results.append(row)
                    with open(ledger_path(pile), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    if verbose:
                        n = len(results)
                        wa = sum(r["winner"] == "a" for r in results)
                        print(f"  game {n:3d}: {ta} vs {tb} -> "
                              f"{row['winner']}   (A {wa}/{n}, "
                              f"{(time.time() - t0) / n:.0f}s/game)")
        finally:
            with lock:
                alive.discard(wid)
            for ap in procs.values():
                ap.close()
            sc.close()

    threads, next_wid = [], 0

    def spawn():
        """Start one more worker thread (caller holds no locks)."""
        nonlocal next_wid
        wid = next_wid
        next_wid += 1
        with lock:
            alive.add(wid)
        t = threading.Thread(target=worker, args=(wid,), daemon=True)
        threads.append(t)
        t.start()

    for _ in range(max(1, workers)):
        spawn()
    # pool manager: grow toward the (possibly dashboard-adjusted) target
    # while jobs remain; shrinking happens inside the workers themselves
    while True:
        with lock:
            n_alive, jobs_left = len(alive), len(jobs)
        if not n_alive:
            break
        if controls and jobs_left and n_alive < controls.target_workers:
            spawn()
        time.sleep(0.5)
    for t in threads:
        t.join()
    if controls:
        controls.detach()
    report(name_a, name_b, results)
    return results


def report(name_a, name_b, results):
    """Print the series score with a Wilson interval and Elo mapping."""
    n = len(results)
    if not n:
        print("no games played")
        return
    wa = sum(r["winner"] == "a" for r in results)
    wb = sum(r["winner"] == "b" for r in results)
    ties = n - wa - wb
    score = (wa + 0.5 * ties) / n
    lo, hi = wilson(wa + 0.5 * ties, n)
    print(f"\n{name_a} vs {name_b}: {wa}-{wb}-{ties} "
          f"({score:.1%}, 95% CI {lo:.1%}-{hi:.1%})  "
          f"elo {elo_diff(score):+.0f} [{elo_diff(lo):+.0f}, "
          f"{elo_diff(hi):+.0f}]")
    for tag, name in (("a", name_a), ("b", name_b)):
        moves = sum(r[f"moves_{tag}"] for r in results)
        wall = sum(r[f"wall_{tag}"] for r in results)
        extras = {k: sum(r[f"{k}_{tag}"] for r in results)
                  for k in ("errors", "late", "illegal")}
        forfeits = sum(r.get("forfeit") == tag for r in results)
        print(f"  {name}: {wall / max(1, moves):.2f}s/move"
              + "".join(f", {k} {v}" for k, v in extras.items() if v)
              + (f", forfeits {forfeits}" if forfeits else ""))


def bt_rows(rows):
    """Bradley-Terry table rows over ledger results, best first — shared by
    the CLI standings printer and the live dashboard panel."""
    import math
    if not rows:
        return []
    players = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    wins = {p: {q: 0.0 for q in players} for p in players}
    info = {p: {"arch": "?", "commit": "", "moves": 0, "wall": 0.0,
                "forfeits": 0} for p in players}
    for r in rows:
        pa, pb = r["a"], r["b"]
        if r["winner"] == "a":
            wins[pa][pb] += 1
        elif r["winner"] == "b":
            wins[pb][pa] += 1
        else:
            wins[pa][pb] += 0.5
            wins[pb][pa] += 0.5
        for tag, p in (("a", pa), ("b", pb)):
            info[p]["arch"] = r.get(f"arch_{tag}", "?")
            info[p]["commit"] = r.get(f"commit_{tag}", "")
            info[p]["moves"] += r.get(f"moves_{tag}", 0)
            info[p]["wall"] += r.get(f"wall_{tag}", 0.0)
            info[p]["forfeits"] += r.get("forfeit") == tag
    rating = {p: 1.0 for p in players}
    for _ in range(200):          # standard BT fixed-point iteration
        for p in players:
            num = sum(wins[p].values())
            den = sum((wins[p][q] + wins[q][p]) / (rating[p] + rating[q])
                      for q in players if q != p)
            if den > 0:
                rating[p] = max(num / den, 1e-9)
        m = sum(rating.values()) / len(rating)
        rating = {p: v / m for p, v in rating.items()}
    out = []
    for p in sorted(players, key=lambda p: -rating[p]):
        i = info[p]
        out.append({"name": p, "rating": 1500 + 400 * math.log10(rating[p]),
                    "arch": i["arch"], "commit": i["commit"],
                    "games": int(sum(wins[p].values())
                                 + sum(wins[q][p] for q in players)),
                    "spm": i["wall"] / max(1, i["moves"]),
                    "forfeits": i["forfeits"]})
    return out


def standings(pile):
    """Print Bradley-Terry ratings over every ledger row, one pool."""
    rows = read_ledger(pile)
    if not rows:
        print(f"no results in {ledger_path(pile)} — run some games first")
        return
    print(f"pile standings ({len(rows)} games):")
    print(f"{'rating':>7s}  {'agent':20s} {'architecture':26s} "
          f"{'s/move':>7s} {'games':>6s}  notes")
    for r in bt_rows(rows):
        notes = (f"commit {r['commit']}"
                 + (f", {r['forfeits']} forfeits" if r["forfeits"] else ""))
        print(f"{r['rating']:7.0f}  {r['name']:20s} {r['arch']:26s} "
              f"{r['spm']:7.2f} {r['games']:6d}  {notes}")


def list_bundles(pile):
    """Print every bundle in the pile with its provenance."""
    bundles = load_pile(pile)
    if not bundles:
        print(f"no bundles in {pile} — export some with export_agent.py")
        return
    print(f"{'name':20s} {'agent':12s} {'architecture':26s} "
          f"{'commit':8s} {'created':11s} notes")
    for name, b in bundles.items():
        m = b["manifest"]
        commit = m.get("git", {}).get("commit", "")
        commit += "*" if m.get("git", {}).get("dirty") else ""
        print(f"{name:20s} {m.get('agent', '?'):12s} "
              f"{m.get('architecture', '?'):26s} {commit:8s} "
              f"{m.get('created', ''):11s} {m.get('notes', '')}")


def main(cfg=CFG):
    """CLI entry: dispatch list/play/star/all/standings on a pile."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    pile = pile_dir(opt("--pile"))
    cmd = args[0]
    if cmd == "list":
        list_bundles(pile)
        return
    if cmd == "standings":
        standings(pile)
        return

    bundles = load_pile(pile)
    workers = int(opt("--workers", 2))
    save = "--no-save" not in args
    kw = dict(cfg=cfg,
              workers=workers,
              quick=int(opt("--quick", 0)) or None,
              repeat=int(opt("--repeat", 1)),
              budget=float(opt("--move-budget", 0)) or None,
              hang_timeout=float(opt("--hang-timeout", 300)),
              temperature=float(opt("--temp", 0.0)),
              label=opt("--label"),
              save_replays=save)
    if cmd not in ("play", "star", "all"):
        print(__doc__)
        return

    def dashboard(run_label, plan_total=1):
        """One live spectator + control panel shared by every pairing.
        --no-live keeps the old headless (still replay-saving) behavior."""
        if "--no-live" in args:
            return None, None
        from spectate import Spectator
        controls = TournamentControls(workers, pile=pile, budget=kw["budget"])
        controls.plan_total = plan_total
        spectator = Spectator(run_label, cfg, live=True,
                              port=int(opt("--dash-port", 8020)), save=save,
                              controls=controls)
        if save:
            print(f"  saving replays under {spectator.dir}/")
        print("  dashboard: game trackers, skip-matchup, worker +/- and "
              "pause live at the URL above")
        return spectator, controls

    if cmd == "play":
        a, b = args[1], args[2]
        for name in (a, b):
            assert name in bundles, f"no bundle '{name}' (round_robin.py list)"
        spectator, controls = dashboard(f"{a}_vs_{b}")
        run_pairing(a, b, pile, bundles, spectator=spectator,
                    controls=controls, **kw)
    elif cmd == "star":
        anchor = opt("--anchor", "baseline")
        assert anchor in bundles, \
            f"anchor bundle '{anchor}' not in the pile — export it first"
        others = [n for n in bundles if n != anchor]
        spectator, controls = dashboard(f"star_{anchor}", len(others))
        for name in others:
            print(f"\n=== {name} vs {anchor} ===")
            run_pairing(name, anchor, pile, bundles, spectator=spectator,
                        controls=controls, **kw)
        standings(pile)
    elif cmd == "all":
        names = sorted(bundles)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        spectator, controls = dashboard("all", len(pairs))
        for a, b in pairs:
            print(f"\n=== {a} vs {b} ===")
            run_pairing(a, b, pile, bundles, spectator=spectator,
                        controls=controls, **kw)
        standings(pile)


if __name__ == "__main__":
    main()
