"""Outcome-driven self-play for the JEPA-Consequence agent (jepa-c).

Why this exists: jepa-c decides ~300x faster than the DUCT baseline because it
never touches the simulator at decision time. That flips the self-play economics
— generation is *sim-bound*, so the same box that fed DUCT a trickle of games
can feed jepa-c orders of magnitude more real-engine games per hour. This loop
turns that throughput into strength.

Why plain BC self-play would NOT work: imitating your own argmax is a fixed
point — the model would just re-learn itself. Every objective here is grounded
in the real game outcome instead:

  value      MSE toward the final result z in {-1, 0, +1} (per side).
  policy     advantage-weighted cross-entropy over the exact candidate set the
             chooser scored at play time: weight = clip(exp((z - b)/beta)),
             where the baseline b is the mean predicted value over that
             decision's candidates (detached). Moves that beat expectation are
             reinforced; moves that lost despite looking fine are suppressed.
             With z > b the weight exceeds 1 even for the model's own choice,
             so the policy *improves* rather than self-imitates.
  jepa       the taken move's consequence vector is matched to the EMA target
             encoding of the *realized* next decision position — the world
             model keeps learning dynamics on fresh, on-policy states.

Pitfall engineering (each of these has burned this kind of loop before):
  * league opponents — each game's opponent is the current model (mirror), a
    random past league checkpoint, or the frozen starting anchor, so strategy
    cycling / opponent overfitting can't hide (spj_p_* knobs).
  * human anchor — a capped fraction of the original human-BC shards is mixed
    into every training iteration, so self-play cannot drift into a private
    meta that forgets how humans punish it.
  * exploration — temperature schedule plus eps-uniform candidate picks during
    generation ONLY (evaluation/gates are argmax, no noise).
  * team diversity — teams are sampled from the sim-validated self-play pool
    (artifacts/selfplay_teams.json, ~3k real tournament/dataset teams), not the
    10 replicas, so pairwise team artifacts can't be memorized.
  * train=play identity — samples come from the chooser's own recorded plan
    (the exact positions, candidate arrays, and choice it acted on), the bug
    class that sank the first exported agent.
  * gating — every iteration plays an argmax series vs the best checkpoint so
    far; best only advances at >= spj_gate_keep. Generation stays on-policy
    (current model), but promotion is earned.

Layout: checkpoints in checkpoint_dir/jepa/selfplay/ (spj_last.pt, spj_best.pt,
spj_anchor.pt, league/spj_iter_NNN.pt), replay buffer in .../selfplay/buffer/.
Resumable via state.json.

CLI: python selfplay_jepa.py [--hours H] [--iters N] [--from ckpt] [--fresh]
                             [--games G] [--procs P] [--workers W] [--no-gate]
"""

import json
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import teams as teams_mod
from config import CFG
from jepa.config import JCFG
from jepa.vocab import JEPAVocab
from models.jepa_consequence import JEPAConsequenceModel
from observe_game import Bot
from train_jepa import _vicreg


def spj_dir(cfg=CFG):
    """Return the jepa-c self-play checkpoint directory ``Path``."""
    return cfg.checkpoint_dir / "jepa" / "selfplay"


def buffer_dir(cfg=CFG):
    """Return the jepa-c self-play replay-buffer directory ``Path``."""
    return spj_dir(cfg) / "buffer"


def league_dir(cfg=CFG):
    """Return the league (past-checkpoint opponents) directory ``Path``."""
    return spj_dir(cfg) / "league"


def team_pool(cfg=CFG):
    """Return ``{name: sets}``: the replica teams plus every pool team.

    The pool file (built by the team tooling) holds ~3k sim-validated real
    teams; falling back to the 10 replicas keeps the script runnable on a
    box that never built it."""
    out = {name: teams_mod.get(name) for name in teams_mod.TEAMS}
    p = cfg.artifacts_dir / "selfplay_teams.json"
    if p.exists():
        for name, entry in json.loads(p.read_text())["teams"].items():
            out[name] = entry["sets"]
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

class RecorderConsequenceBot(Bot):
    """CTS-honest Bot that records the chooser's exact plan at each decision.

    Adds eps-uniform exploration over the chooser's own candidate list
    (generation only), re-deriving the choice string so the recorded chosen
    index always matches the move actually sent to the sim."""

    def __init__(self, *args, eps=0.0, rng=None, n_cand=12, **kw):
        """Wrap ``Bot`` with exploration knobs and an empty sample list."""
        super().__init__(*args, **kw)
        self.eps = eps
        self.exp_rng = rng or random.Random(0)
        self.n_cand = n_cand
        self.samples = []          # per-decision dicts; z paired at game end

    def decide(self, request, temperature):
        """Choose (with recording + exploration) and append one sample."""
        from actions import joint_choice
        self.belief.update(self.tracker.drain_events(), viewer=self.side)
        ch = self.searcher
        ch.record, ch.last_plan = True, None
        try:
            joint, info = ch.choose(self.tracker, self.belief, self.side,
                                    request, self.brought,
                                    temperature=temperature)
        finally:
            ch.record = False
        plan = ch.last_plan
        ch.last_plan = None
        if plan is not None:
            idx = plan["chosen"]
            if self.eps > 0 and len(plan["cands"]) > 1 \
                    and self.exp_rng.random() < self.eps:
                idx = self.exp_rng.randrange(len(plan["cands"]))
                joint = plan["cands"][idx]
            self.samples.append(self._sample(plan, idx))
        return joint_choice(request, joint, self.name_to_idx), info

    def _sample(self, plan, idx):
        """Build one shard-format sample from a recorded plan + chosen index."""
        pos = plan["pos"]
        acts = plan["cand_acts"]
        rest = [j for j in range(len(acts)) if j != idx]
        self.exp_rng.shuffle(rest)
        order = ([idx] + rest)[:self.n_cand]
        cand_acts = np.zeros((self.n_cand, 12, 7), dtype=np.int16)
        cand_mask = np.zeros(self.n_cand, dtype=bool)
        for row, j in enumerate(order):
            cand_acts[row] = acts[j]
            cand_mask[row] = True
        return {
            "cur_gcat": pos.global_cat.astype(np.int16),
            "cur_gscal": pos.global_scalar,
            "cur_mcat": pos.mon_cat.astype(np.int16),
            "cur_mscal": pos.mon_scalar,
            "cur_dmg": pos.dmg_edge,
            "my_act": acts[idx].astype(np.int16),
            "cand_acts": cand_acts, "cand_mask": cand_mask,
            "a_index": np.int16(0),
        }


def _pair_and_flush(bot, z, out):
    """Pair consecutive decisions into (cur -> realized next) rows.

    The JEPA target for decision t is the bot's own position at decision t+1 —
    the realized future after the opponent responded and chance resolved. The
    game's last decision has no future; it still trains value/policy with
    ``has_nxt=0`` masking its jepa loss."""
    n = len(bot.samples)
    for i, s in enumerate(bot.samples):
        nxt = bot.samples[i + 1] if i + 1 < n else None
        row = dict(s)
        for key in ("gcat", "gscal", "mcat", "mscal", "dmg"):
            row[f"nxt_{key}"] = (nxt[f"cur_{key}"] if nxt
                                 else np.zeros_like(s[f"cur_{key}"]))
        row["has_nxt"] = np.bool_(nxt is not None)
        row["value"] = np.int8(z)
        row["weight"] = np.float32(1.0)
        for k, v in row.items():
            out[k].append(v)


def _make_chooser(model, vocab, cfg, seed):
    """Build one thread-local chooser around a shared loaded model."""
    from agents.jepa_world_model.v2 import JEPAConsequenceChooser
    bridge = None
    if getattr(model.jcfg, "use_damage_features", False):
        from damage import DamageBridge
        bridge = DamageBridge(cfg)
    return JEPAConsequenceChooser(model, vocab, cfg, model.jcfg, seed, bridge)


def _load_model(path, device, cache):
    """Load (and cache) one consequence checkpoint for opponent sampling."""
    key = str(path)
    if key not in cache:
        m, _ = JEPAConsequenceModel.load(path, device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        cache[key] = m
    return cache[key]


def _sample_opponent(rng, jcfg, cfg):
    """Pick this game's opponent checkpoint path (None = mirror)."""
    r = rng.random()
    if r < jcfg.spj_p_mirror:
        return None
    league = sorted(league_dir(cfg).glob("*.pt"))
    if r < jcfg.spj_p_mirror + jcfg.spj_p_league and league:
        return rng.choice(league)
    anchor = spj_dir(cfg) / "spj_anchor.pt"
    return anchor if anchor.exists() else None


def generate_games(cfg, jcfg, n_games, workers, seed, verbose=True):
    """Play ``n_games`` self-play games on ``workers`` threads; return arrays.

    Threads each own a Sidecar + choosers (own damage bridges — the bridge
    pipe protocol is not thread-safe); the current model and any sampled
    opponent models are shared across threads (inference-only)."""
    from beliefs import load_dex
    from env import Sidecar, random_choice
    from selfplay import play_selfplay_game

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    current = _load_model(spj_dir(cfg) / "spj_last.pt", device, cache)
    vocab = (JEPAVocab.from_state(current.vocab_state, load_dex(cfg))
             if current.vocab_state else JEPAVocab.build(cfg))
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    pool = team_pool(cfg)
    names = list(pool)
    # play_selfplay_game reads the temperature schedule off cfg.sp_* fields
    import dataclasses
    gen_cfg = dataclasses.replace(cfg, sp_temp_turns=jcfg.spj_temp_turns,
                                  sp_final_temp=jcfg.spj_final_temp)

    jobs = list(range(n_games))
    lock = threading.Lock()
    keys = ("cur_gcat", "cur_gscal", "cur_mcat", "cur_mscal", "cur_dmg",
            "my_act", "cand_acts", "cand_mask", "a_index",
            "nxt_gcat", "nxt_gscal", "nxt_mcat", "nxt_mscal", "nxt_dmg",
            "has_nxt", "value", "weight")
    out = {k: [] for k in keys}
    done = [0]
    t0 = time.time()

    def worker(wid):
        rng = random.Random((seed << 16) + wid)
        sc = Sidecar(cfg)
        cur_chooser = _make_chooser(current, vocab, cfg, (seed << 16) + wid)
        opp_choosers = {}          # ckpt path -> chooser (thread-local bridges)
        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    jobs.pop()
                opp_path = _sample_opponent(rng, jcfg, cfg)
                if opp_path is None:
                    opp_chooser = cur_chooser
                else:
                    if str(opp_path) not in opp_choosers:
                        with lock:      # torch.load once per path per proc
                            m = _load_model(opp_path, device, cache)
                        opp_choosers[str(opp_path)] = _make_chooser(
                            m, vocab, cfg, (seed << 17) + wid)
                    opp_chooser = opp_choosers[str(opp_path)]
                cur_side = "p1" if rng.random() < 0.5 else "p2"
                opp_side = "p2" if cur_side == "p1" else "p1"
                sets = {s: pool[rng.choice(names)] for s in ("p1", "p2")}
                bots = {
                    cur_side: RecorderConsequenceBot(
                        cur_side, sets[cur_side], sets[opp_side], cur_chooser,
                        usage, cfg, eps=jcfg.spj_eps, rng=rng,
                        n_cand=jcfg.n_cand),
                    opp_side: RecorderConsequenceBot(
                        opp_side, sets[opp_side], sets[cur_side], opp_chooser,
                        usage, cfg, eps=jcfg.spj_eps, rng=rng,
                        n_cand=jcfg.n_cand),
                }
                winner = play_selfplay_game(sc, bots, sets, gen_cfg, rng,
                                            max_turns=jcfg.spj_max_turns)
                with lock:
                    # record BOTH sides in mirror games, only OUR side vs league
                    for s, o in (("p1", "p2"), ("p2", "p1")):
                        if opp_path is not None and s == opp_side:
                            continue
                        z = 1 if winner == s else -1 if winner == o else 0
                        _pair_and_flush(bots[s], z, out)
                    done[0] += 1
                    if verbose and done[0] % 20 == 0:
                        r = (time.time() - t0) / done[0]
                        print(f"  {done[0]}/{n_games} games, {r:.1f}s/game, "
                              f"{len(out['value'])} samples", flush=True)
        finally:
            sc.close()
            cur_chooser.close()
            for ch in opp_choosers.values():
                ch.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    casts = {"cur_gcat": np.int16, "cur_mcat": np.int16, "nxt_gcat": np.int16,
             "nxt_mcat": np.int16, "my_act": np.int16, "cand_acts": np.int16,
             "a_index": np.int16, "value": np.int8, "weight": np.float32,
             "has_nxt": bool, "cand_mask": bool}
    return {k: np.stack(v).astype(casts.get(k, np.float32))
            for k, v in out.items()}


def _gen_subprocess(cfg=CFG, jcfg=JCFG):
    """Entry for ``--_gen <iter> <proc> <games> <seed> <workers>``."""
    it, proc, games, seed, workers = (int(x) for x in sys.argv[2:7])
    arrs = generate_games(cfg, jcfg, games, workers, seed, verbose=proc == 0)
    assert arrs["value"].size, "generator produced no samples"
    np.savez_compressed(buffer_dir(cfg) / f"spj_{it:04d}_{proc}.npz", **arrs)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _load_buffer(cur_iter, cfg, jcfg):
    """Concatenate the rolling self-play buffer window into one array dict."""
    lo = max(0, cur_iter - jcfg.spj_buffer_iters + 1)
    files = [f for f in sorted(buffer_dir(cfg).glob("spj_*.npz"))
             if lo <= int(f.stem.split("_")[1]) <= cur_iter]
    assert files, "empty self-play buffer"
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def _mix_human(arr, cfg, jcfg, rng):
    """Blend a capped sample of human-BC rows into the buffer arrays.

    The human anchor stops self-play from drifting into a private meta; rows
    are drawn fresh each iteration from a random human shard, capped so they
    are ``spj_human_mix`` of the final mixture."""
    shards = sorted((cfg.artifacts_dir / "jepa_cons_prepped").glob("train_*.npz"))
    if not shards or jcfg.spj_human_mix <= 0:
        return arr
    h = np.load(rng.choice(shards))
    n_sp = len(arr["value"])
    n_h = min(len(h["value"]),
              int(n_sp * jcfg.spj_human_mix / max(1e-6, 1 - jcfg.spj_human_mix)))
    idx = rng.choice(len(h["value"]), size=n_h, replace=False) if n_h else []
    if not len(idx):
        return arr
    merged = {}
    for k in arr:
        if k in h.files:
            merged[k] = np.concatenate([arr[k], h[k][idx]])
        elif k == "has_nxt":       # human prep rows always carry a real future
            merged[k] = np.concatenate([arr[k], np.ones(n_h, dtype=bool)])
        else:
            raise KeyError(f"human shard missing key {k}")
    return merged


def sp_losses(model, sh, jcfg):
    """Advantage-weighted self-play loss for one minibatch tensor dict."""
    cur = {"gcat": sh["cur_gcat"], "gscal": sh["cur_gscal"],
           "mcat": sh["cur_mcat"], "mscal": sh["cur_mscal"],
           "dmg": sh["cur_dmg"]}
    nxt = {"gcat": sh["nxt_gcat"], "gscal": sh["nxt_gscal"],
           "mcat": sh["nxt_mcat"], "mscal": sh["nxt_mscal"],
           "dmg": sh["nxt_dmg"]}
    z = model.encode(cur)
    b, _, d = z.shape

    # jepa: consequence of the taken move vs realized future (masked)
    c_true = model.consequence(z, sh["my_act"], cur["dmg"], None)
    target = model.target_context(nxt)
    per = F.smooth_l1_loss(c_true, target, reduction="none").mean(-1)
    mask = sh["has_nxt"].float()
    jepa_l = (per * mask).sum() / mask.sum().clamp_min(1.0)

    # value: outcome regression on the taken move's consequence
    v = model.value(c_true)
    value_l = F.mse_loss(v, sh["value"])

    # policy: advantage-weighted CE over the recorded candidate set
    nc = sh["cand_acts"].shape[1]
    z_rep = z.unsqueeze(1).expand(b, nc, z.shape[1], d).reshape(b * nc, -1, d)
    dmg_rep = cur["dmg"].unsqueeze(1).expand(b, nc, 6, 6).reshape(b * nc, 6, 6)
    c_cand = model.consequence(z_rep, sh["cand_acts"].reshape(b * nc, 12, 7),
                               dmg_rep, None)
    logits = model.score(c_cand).reshape(b, nc)
    logits = logits.masked_fill(~sh["cand_mask"], float("-inf"))
    with torch.no_grad():
        v_cand = model.value(c_cand).reshape(b, nc)
        v_cand = v_cand.masked_fill(~sh["cand_mask"], 0.0)
        baseline = v_cand.sum(-1) / sh["cand_mask"].sum(-1).clamp_min(1)
        adv = sh["value"] - baseline
        w = torch.exp(adv / jcfg.spj_beta).clamp(max=jcfg.spj_w_max)
    ce = F.cross_entropy(logits, sh["a_index"], reduction="none")
    policy_l = (w * ce).mean()

    var_l, cov_l = _vicreg(z, jcfg.vicreg_gamma)
    total = (jcfg.w_jepa_c * jepa_l + jcfg.w_value_c * value_l
             + jcfg.w_bc * policy_l
             + jcfg.w_vicreg_var * var_l + jcfg.w_vicreg_cov * cov_l)
    acc = (logits.argmax(-1) == sh["a_index"]).float().mean()
    return total, {"total": total.item(), "jepa": jepa_l.item(),
                   "value": value_l.item(), "policy": policy_l.item(),
                   "acc": acc.item(), "adv_w": w.mean().item()}


def train_iteration(model, cur_iter, device, cfg, jcfg, rng):
    """Fine-tune ``model`` in place on buffer + human mix; return avg metrics."""
    arr = _mix_human(_load_buffer(cur_iter, cfg, jcfg), cfg, jcfg, rng)
    n = len(arr["value"])
    opt = torch.optim.AdamW(model.parameters(), lr=jcfg.spj_lr,
                            weight_decay=jcfg.weight_decay)
    long_keys = {"cur_gcat", "cur_mcat", "nxt_gcat", "nxt_mcat", "my_act",
                 "cand_acts", "a_index"}
    model.train()
    agg = {}
    for ep in range(jcfg.spj_epochs):
        order = rng.permutation(n)
        for i in range(0, n, jcfg.batch_size):
            idx = order[i:i + jcfg.batch_size]
            sh = {}
            for k, a in arr.items():
                t = torch.as_tensor(
                    a[idx].astype(np.int64 if k in long_keys else
                                  bool if k in ("cand_mask", "has_nxt")
                                  else np.float32), device=device)
                sh[k] = t
            loss, m = sp_losses(model, sh, jcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), jcfg.grad_clip)
            opt.step()
            model.update_ema()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["_n"] = agg.get("_n", 0) + 1
    model.eval()
    return {k: agg[k] / agg["_n"] for k in agg if k != "_n"}


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def gate(cfg, jcfg, n_games, workers=4):
    """Argmax, no-noise series: spj_last vs spj_best. Returns last's score."""
    import dataclasses

    from beliefs import load_dex
    from env import Sidecar
    from selfplay import play_selfplay_game

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    new = _load_model(spj_dir(cfg) / "spj_last.pt", device, cache)
    best = _load_model(spj_dir(cfg) / "spj_best.pt", device, cache)
    vocab = (JEPAVocab.from_state(new.vocab_state, load_dex(cfg))
             if new.vocab_state else JEPAVocab.build(cfg))
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    pool = team_pool(cfg)
    names = list(pool)
    argmax_cfg = dataclasses.replace(cfg, sp_temp_turns=0, sp_final_temp=0.0)
    jobs = list(range(n_games))
    lock, score = threading.Lock(), [0.0]

    def worker(wid):
        rng = random.Random(9000 + wid)
        sc = Sidecar(cfg)
        cn = _make_chooser(new, vocab, cfg, wid)
        cb = _make_chooser(best, vocab, cfg, 1000 + wid)
        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    g = jobs.pop()
                new_side = "p1" if g % 2 == 0 else "p2"
                old_side = "p2" if new_side == "p1" else "p1"
                sets = {s: pool[rng.choice(names)] for s in ("p1", "p2")}
                bots = {new_side: Bot(new_side, sets[new_side],
                                      sets[old_side], cn, usage, cfg),
                        old_side: Bot(old_side, sets[old_side],
                                      sets[new_side], cb, usage, cfg)}
                w = play_selfplay_game(sc, bots, sets, argmax_cfg, rng,
                                       max_turns=jcfg.spj_max_turns)
                with lock:
                    score[0] += (1.0 if w == new_side
                                 else 0.5 if w is None else 0.0)
        finally:
            sc.close()
            cn.close()
            cb.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return score[0] / max(1, n_games)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(cfg=CFG, jcfg=JCFG):
    """Run resumable generate -> train -> gate iterations from CLI flags."""
    if len(sys.argv) > 1 and sys.argv[1] == "--_gen":
        _gen_subprocess(cfg, jcfg)
        return
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    hours = float(opt("--hours", 0)) or None
    iters = int(opt("--iters", 0)) or None
    games = int(opt("--games", jcfg.spj_games_per_iter))
    procs = int(opt("--procs", jcfg.spj_procs))
    if opt("--workers"):
        jcfg.spj_workers = int(opt("--workers"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in (spj_dir(cfg), buffer_dir(cfg), league_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
    state_p = spj_dir(cfg) / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else \
        {"iter": -1, "history": []}

    last = spj_dir(cfg) / "spj_last.pt"
    if last.exists() and "--fresh" not in args:
        model, _ = JEPAConsequenceModel.load(last, device)
        print(f"resumed jepa-c self-play at iteration {state['iter'] + 1}")
    else:
        stale = list(buffer_dir(cfg).glob("spj_*.npz"))
        assert not stale, \
            f"--fresh with {len(stale)} old buffer shards — move them first"
        src = Path(opt("--from",
                       cfg.checkpoint_dir / "jepa" / "jepa_consequence.pt"))
        model, _ = JEPAConsequenceModel.load(src, device)
        model.save(last)
        model.save(spj_dir(cfg) / "spj_best.pt")
        model.save(spj_dir(cfg) / "spj_anchor.pt")
        state = {"iter": -1, "history": []}
        print(f"forked {src} -> {last} (also best + frozen anchor)")
    nrng = np.random.default_rng(0)

    deadline = time.time() + hours * 3600 if hours else None
    start_iter = state["iter"] + 1
    it = state["iter"]
    while True:
        it += 1
        if iters is not None and it >= start_iter + iters:
            break
        if deadline and time.time() > deadline:
            print("time budget reached")
            break
        t0 = time.time()
        print(f"\n=== iteration {it} "
              f"({datetime.now().strftime('%H:%M')}) ===", flush=True)

        per = [games // procs + (i < games % procs) for i in range(procs)]
        cmds = [subprocess.Popen(
            [sys.executable, __file__, "--_gen", str(it), str(i),
             str(per[i]), str(it * 131 + i), str(jcfg.spj_workers)])
            for i in range(procs) if per[i] > 0]
        codes = [c.wait() for c in cmds]
        assert all(c == 0 for c in codes), f"generator failed: {codes}"
        n_new = sum(len(np.load(f)["value"])
                    for f in buffer_dir(cfg).glob(f"spj_{it:04d}_*.npz"))
        print(f"  generated {games} games / {n_new} samples "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)

        metrics = train_iteration(model, it, device, cfg, jcfg, nrng)
        print("  train: " + " ".join(f"{k} {v:.4f}"
                                     for k, v in metrics.items()), flush=True)
        model.save(last)
        if it % jcfg.spj_league_every == 0:
            model.save(league_dir(cfg) / f"spj_iter_{it:03d}.pt")

        g = None
        if "--no-gate" not in args and jcfg.spj_gate_games > 0:
            g = gate(cfg, jcfg, jcfg.spj_gate_games)
            print(f"  gate vs best: {g:.0%} ({jcfg.spj_gate_games} games)",
                  flush=True)
            if g >= jcfg.spj_gate_keep:
                model.save(spj_dir(cfg) / "spj_best.pt")
                print("  promoted -> spj_best.pt", flush=True)

        state["iter"] = it
        state["history"].append({
            "iter": it, "games": games, "samples": n_new, "gate": g,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "minutes": round((time.time() - t0) / 60, 1),
            "date": datetime.now().isoformat(timespec="minutes")})
        state_p.write_text(json.dumps(state, indent=1))
    print("\ndone — checkpoints in", spj_dir(cfg))
    print("export the winner:  python export_agent.py exp-jepa-c-sp "
          "--agent jepa-c --ckpt", spj_dir(cfg) / "spj_best.pt",
          "--architecture JEPA-Consequence")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
    else:
        main()
