"""AlphaZero-style self-play training, forked from the behavior-cloned
predictor. The BC pipeline (data.py / train.py) is untouched; this file owns
everything self-play.

The analog to AlphaZero, adapted to simultaneous turns and hidden information:

  policy target  the root visit distribution of the DUCT search (a mixed
                 strategy over JOINT actions — search output is inherently
                 joint, which is why self-play requires the joint policy
                 head). Aggregated across determinizations, it is the
                 marginal over the bot's own belief about hidden sets.
  value target   the final game outcome z in {-1, 0, +1} from each player's
                 perspective (a tie, incl. the stall cap, is 0).
  exploration    Dirichlet noise on the root priors + temperature sampling
                 for the first sp_temp_turns turns — both OFF outside
                 generation, so play/benchmark behavior is unchanged.
  games          CTS-honest: each side sees its own team plus reveals, and
                 runs its real tracker + particle filter (same as
                 observe_game). Training inputs are encoded from the REAL
                 belief summary, exactly like data.py prep — so BC shards and
                 self-play shards are the same language. Oracle aux labels
                 come from the true opponent sets held by the generator.

Parallelism ("on the GPU" as far as physics allows): the simulator is the
real Showdown engine on CPU, so the GPU's job is inference. Generation runs
sp_procs subprocesses (sidestepping the GIL) x sp_workers game threads, and
every leaf evaluation in every game funnels through one BatchedEvaluator per
process — a queue that coalesces requests from all threads into single
batched predict_batch calls. The evaluator quacks like the model (only
predict_batch is ever called), so the versioned chooser needs no special
self-play code.

Iteration loop: generate sp_games_per_iter games -> append a shard to the
replay buffer -> fine-tune on the last sp_buffer_iters shards -> checkpoint ->
quick gate vs the previous iteration. Checkpoints live in
checkpoint_dir/selfplay/ (sp_last.pt, sp_iter_NNN.pt) and never touch the BC
checkpoints. Archive milestones with `python benchmark.py archive`.

CLI: python selfplay.py [--hours H] [--iters N] [--from ckpt] [--fresh]
                        [--games G] [--procs P] [--workers W] [--sims S]
                        [--no-gate]
Forking a v1 (per-slot) checkpoint converts it with PolicyValueNet.from_slot
— the joint head starts as the exact factorized distribution and learns the
correlations from there.
"""

import json
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import teams as teams_mod
from actions import to_index
from agents.determinized_duct.v1 import DeterminizedDUCTChooser
from config import CFG
from damage import damage_features
from env import Sidecar, SidecarBattle, pack_team, random_choice
from models.policy_value import PolicyValueNet
from observe_game import Bot
from search.mcts import joint_choice
from tokenizer import PositionTokenizer
from train import make_loader


def sp_dir(cfg=CFG):
    """Return the self-play checkpoint directory ``Path``."""
    return cfg.checkpoint_dir / "selfplay"


def buffer_dir(cfg=CFG):
    """Return the self-play replay-buffer directory ``Path``."""
    return sp_dir(cfg) / "buffer"


# ---------------------------------------------------------------------------
# batched GPU evaluation across game threads
# ---------------------------------------------------------------------------

class BatchedEvaluator:
    """Coalesces predict_batch calls from many game threads into single GPU
    batches. Passed to ``DeterminizedDUCTChooser`` as its model dependency;
    ``PolicyValueLeafEvaluator`` only calls ``predict_batch``, so search cannot
    distinguish this queue from a direct ``PolicyValueNet``."""

    def __init__(self, model, max_batch=256, wait_ms=2.0):
        """Start a daemon queue worker around a predict-batch model."""
        self.model = model
        self.q = queue.Queue()
        self.max_batch = max_batch
        self.wait_s = wait_ms / 1000.0
        self.stats = {"calls": 0, "rows": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def predict_batch(self, tokens):
        """Synchronously return the standard NumPy ``ModelPrediction`` tuple."""
        ev, out = threading.Event(), {}
        self.q.put((np.asarray(tokens), ev, out))
        ev.wait()
        return out["r"]

    def _loop(self):
        """Forever coalesce queued arrays and fulfill their events in order."""
        while True:
            batch = [self.q.get()]                     # block for the first
            deadline = time.monotonic() + self.wait_s
            rows = len(batch[0][0])
            while rows < self.max_batch:
                try:
                    item = self.q.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                batch.append(item)
                rows += len(item[0])
            toks = np.concatenate([b[0] for b in batch])
            dists, values, aux = self.model.predict_batch(toks)
            self.stats["calls"] += 1
            self.stats["rows"] += rows
            i = 0
            for t, ev, out in batch:
                j = i + len(t)
                out["r"] = (dists[i:j], values[i:j],
                            {k: v[i:j] for k, v in aux.items()})
                i = j
                ev.set()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

class RecorderBot(Bot):
    """Bot that records (tokens, visit-distribution) at every searched
    decision, encoded from its REAL belief state — the same encoding prep
    uses, so self-play samples speak the tokenizer's language."""

    def __init__(self, *args, noise=None, **kw):
        """Initialize ``Bot`` plus optional ``(epsilon,alpha)`` root noise."""
        super().__init__(*args, **kw)
        self.noise = noise
        self.samples = []          # dicts, z filled in at game end

    def decide(self, request, temperature):
        """Return ``(Showdown choice, ChoiceInfo)`` and append one sample."""
        self.belief.update(self.tracker.drain_events(), viewer=self.side)
        joint, info = self.searcher.choose(
            self.tracker, self.belief, self.side, request, self.brought,
            temperature=temperature, root_noise=self.noise)
        state = self.tracker._view(self.side)
        dmg = damage_features(state, self.belief, self.searcher.bridge) \
            if self.searcher.bridge else {}
        toks = self.searcher.tok.encode(state, self.belief.summary(), dmg)
        self.samples.append({"tokens": toks, "visits": info["visits"],
                             "act": (to_index(joint[0]), to_index(joint[1]))})
        return joint_choice(request, joint, self.name_to_idx), info


def play_selfplay_game(sc, bots, sets_by_side, cfg, rng, max_turns=300,
                       feed=None):
    """Same skeleton as benchmark.run_game, plus the temperature schedule.
    Optional `feed` (a spectate.GameFeed) streams the game to the live
    dashboard + replay saver, so training self-play is watchable too."""
    b = SidecarBattle.create(sc, cfg.format_id,
                             pack_team(sets_by_side["p1"]),
                             pack_team(sets_by_side["p2"]))
    if feed:
        feed.feed(b.log)
    for bot in bots.values():
        bot.feed(b.log)
    turns = 0
    while not b.ended:
        choices = {}
        for side in b.pending_sides():
            req, bot = b.requests[side], bots[side]
            if req.get("teamPreview"):
                n = min(req.get("maxChosenTeamSize") or 4,
                        len(sets_by_side[side]))
                bot.brought = list(range(n))
                choices[side] = "team " + "".join(str(i + 1) for i in range(n))
            elif req.get("forceSwitch"):
                choices[side] = random_choice(req, rng)
            else:
                temp = 1.0 if bot.tracker.turn_no <= cfg.sp_temp_turns \
                    else cfg.sp_final_temp
                choices[side], _ = bot.decide(req, temp)
        resp = b.step(choices)
        if feed:
            feed.feed(resp["log"])
        for bot in bots.values():
            bot.feed(resp["log"])
        if resp["errors"]:
            resp = b.step({s: "default" for s in resp["errors"]})
            if feed:
                feed.feed(resp["log"])
            for bot in bots.values():
                bot.feed(resp["log"])
        turns += 1
        if turns >= max_turns:
            break
    winner = b.winner if b.ended else None
    if feed:
        feed.finish(winner)
    b.destroy()
    return winner


def oracle_labels(tok, opp_sets):
    """Map true opponent sets to item/ability/move auxiliary label lists."""
    items = [tok.item_idx(s["item"]) for s in opp_sets]
    abils = [tok.ability_idx(s["ability"]) for s in opp_sets]
    moves = [[tok.move_idx(m) for m in s["moves"]] + [0] * (4 - len(s["moves"]))
             for s in opp_sets]
    return items, abils, moves


def pad6(labels, fill):
    """Pad/truncate a label list to exactly six entries."""
    return (labels + [fill] * 6)[:6]


def generate_games(model, tok, cfg, n_games, workers, seed, verbose=True):
    """Generate n_games self-play games on `workers` threads sharing one
    BatchedEvaluator. Returns a dict of stacked sample arrays."""
    evaluator = BatchedEvaluator(model)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    team_names = list(teams_mod.TEAMS)
    team_sets = {t: teams_mod.get(t) for t in team_names}
    noise = (cfg.sp_dirichlet_eps, cfg.sp_dirichlet_alpha)
    jobs = list(range(n_games))
    lock = threading.Lock()
    out = {k: [] for k in ("tokens", "pol_idx", "pol_p", "value", "weight",
                           "acts", "opp_items", "opp_abils", "opp_moves")}
    done = [0]
    t0 = time.time()

    def flush(bot, z, opp_sets):
        it, ab, mv = oracle_labels(tok, opp_sets)
        it, ab = pad6(it, 0), pad6(ab, 0)
        mv = pad6(mv, [0, 0, 0, 0])
        K = cfg.sp_policy_targets_k
        for s in bot.samples:
            vis = sorted(s["visits"], key=lambda kv: -kv[1])[:K]
            tot = sum(v for _, v in vis)
            if tot <= 0:
                continue
            idx = [k for k, _ in vis] + [0] * (K - len(vis))
            p = [v / tot for _, v in vis] + [0.0] * (K - len(vis))
            out["tokens"].append(s["tokens"])
            out["pol_idx"].append(idx)
            out["pol_p"].append(p)
            out["value"].append(z)
            out["weight"].append(1.0)
            out["acts"].append(list(s["act"]))
            out["opp_items"].append(it)
            out["opp_abils"].append(ab)
            out["opp_moves"].append(mv)

    def worker(wid):
        rng = random.Random((seed << 16) + wid)
        sc = Sidecar(cfg)
        # one shared sidecar for the game AND this worker's search (was 2)
        searcher = DeterminizedDUCTChooser(
            evaluator, tok, cfg, seed=(seed << 16) + wid, sidecar=sc)
        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    jobs.pop()
                ta, tb = rng.choice(team_names), rng.choice(team_names)
                sets = {"p1": team_sets[ta], "p2": team_sets[tb]}
                bots = {s: RecorderBot(s, sets[s], sets[o], searcher, usage,
                                       cfg, noise=noise)
                        for s, o in (("p1", "p2"), ("p2", "p1"))}
                winner = play_selfplay_game(sc, bots, sets, cfg, rng)
                with lock:
                    for s, o in (("p1", "p2"), ("p2", "p1")):
                        z = 1 if winner == s else -1 if winner == o else 0
                        flush(bots[s], z, sets[o])
                    done[0] += 1
                    if verbose and done[0] % 10 == 0:
                        r = (time.time() - t0) / done[0]
                        print(f"  {done[0]}/{n_games} games, {r:.0f}s/game, "
                              f"{len(out['tokens'])} samples, GPU batch avg "
                              f"{evaluator.stats['rows'] / max(1, evaluator.stats['calls']):.1f}",
                              flush=True)
        finally:
            sc.close()
            searcher.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {
        "tokens": np.array(out["tokens"], dtype=np.uint16),
        "pol_idx": np.array(out["pol_idx"], dtype=np.int16),
        "pol_p": np.array(out["pol_p"], dtype=np.float32),
        "value": np.array(out["value"], dtype=np.int8),
        "weight": np.array(out["weight"], dtype=np.float32),
        "acts": np.array(out["acts"], dtype=np.int8),
        "opp_items": np.array(out["opp_items"], dtype=np.int16),
        "opp_abils": np.array(out["opp_abils"], dtype=np.int16),
        "opp_moves": np.array(out["opp_moves"], dtype=np.int16),
    }


def _gen_subprocess(cfg=CFG):
    """Entry for `python selfplay.py --_gen <iter> <proc> <games> <seed>`:
    load sp_last.pt, generate, write a buffer shard fragment."""
    it, proc, games, seed = (int(x) for x in sys.argv[2:6])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyValueNet.load(sp_dir(cfg) / "sp_last.pt", cfg, device)
    tok = PositionTokenizer.load(cfg)
    arrs = generate_games(model, tok, cfg, games, cfg.sp_workers, seed,
                          verbose=proc == 0)
    assert arrs["value"].size, "generator produced no samples"
    np.savez_compressed(buffer_dir(cfg) / f"sp_{it:04d}_{proc}.npz", **arrs)


# ---------------------------------------------------------------------------
# training on the replay buffer
# ---------------------------------------------------------------------------

class SelfPlayShards(torch.utils.data.Dataset):
    """Batched-index dataset over the last sp_buffer_iters iterations of
    shards (same batched-__getitem__ trick as train.Shards)."""

    def __init__(self, cur_iter, cfg=CFG):
        """Load and concatenate the configured rolling iteration window."""
        lo = max(0, cur_iter - cfg.sp_buffer_iters + 1)
        files = [f for f in sorted(buffer_dir(cfg).glob("sp_*.npz"))
                 if lo <= int(f.stem.split("_")[1]) <= cur_iter]
        assert files, "empty replay buffer"
        parts = [np.load(f) for f in files]
        self.arr = {k: np.concatenate([p[k] for p in parts])
                    for k in parts[0].files}

    def __len__(self):
        """Return replay-buffer position count."""
        return len(self.arr["tokens"])

    def __getitem__(self, idxs):
        """Return the seven-tensor sparse-policy self-play batch."""
        a = self.arr
        return (torch.from_numpy(a["tokens"][idxs].astype(np.int64)),
                torch.from_numpy(a["pol_idx"][idxs].astype(np.int64)),
                torch.from_numpy(a["pol_p"][idxs].astype(np.float32)),
                torch.from_numpy(a["value"][idxs].astype(np.float32)),
                torch.from_numpy(a["opp_items"][idxs].astype(np.int64)),
                torch.from_numpy(a["opp_abils"][idxs].astype(np.int64)),
                torch.from_numpy(a["opp_moves"][idxs].astype(np.int64)))


def sp_loss(model, batch, cfg=CFG):
    """Soft-target CE on the joint head + value MSE + the same aux losses as
    BC (oracle labels are exact in self-play)."""
    tokens, pol_idx, pol_p, value_t, items_t, abils_t, moves_t = batch
    pol, value, (item_lg, abil_lg, move_lg) = model(tokens)
    logp = F.log_softmax(pol.masked_fill(~model.joint_mask, float("-inf")), -1)
    policy_loss = -(pol_p * logp.gather(1, pol_idx)).sum(-1).mean()
    value_loss = F.mse_loss(value, value_t)

    item_loss = F.cross_entropy(item_lg.flatten(0, 1), items_t.flatten(),
                                ignore_index=0)
    abil_loss = F.cross_entropy(abil_lg.flatten(0, 1), abils_t.flatten(),
                                ignore_index=0)
    moves_hot = torch.zeros_like(move_lg).scatter_(-1, moves_t, 1.0)
    moves_hot[..., 0] = 0
    move_loss = F.binary_cross_entropy_with_logits(move_lg, moves_hot)

    loss = (policy_loss + cfg.value_loss_weight * value_loss
            + cfg.aux_set_loss_weight * (item_loss + abil_loss + move_loss))
    return loss, {"loss": loss.detach(), "policy": policy_loss.detach(),
                  "value": value_loss.detach()}


def train_iteration(model, cur_iter, device, cfg=CFG):
    """Fine-tune ``model`` in place on the rolling replay buffer; return None."""
    ds = SelfPlayShards(cur_iter, cfg)
    dl = make_loader(ds, cfg.batch_size, True, device, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sp_lr,
                            weight_decay=cfg.weight_decay,
                            fused=device == "cuda")
    autocast = torch.autocast(device, dtype=torch.bfloat16,
                              enabled=device == "cuda")
    model.train()
    for ep in range(cfg.sp_epochs_per_iter):
        agg, n = {}, 0
        for batch in dl:
            batch = [t.to(device, non_blocking=True) for t in batch]
            for t in batch:
                torch._dynamo.mark_dynamic(t, 0)
            with autocast:
                loss, stats = sp_loss(model, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            n += len(batch[0])
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v * len(batch[0])
        print(f"  train ep{ep}: " + " ".join(
            f"{k} {float(v / n):.4f}" for k, v in agg.items()), flush=True)
    model.eval()


# ---------------------------------------------------------------------------
# gate: quick series new vs previous iteration
# ---------------------------------------------------------------------------

def gate(model_new, model_old, tok, cfg, n_games, workers=4):
    """Argmax, no-noise games between two checkpoints on random replica-team
    pairings. Coarse (n_games is small) — the real verdict is a full
    benchmark.py series against an archived bundle."""
    from benchmark import run_game
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    team_names = list(teams_mod.TEAMS)
    team_sets = {t: teams_mod.get(t) for t in team_names}
    jobs = list(range(n_games))
    lock, score = threading.Lock(), [0.0]

    def worker(wid):
        sc = Sidecar(cfg)
        # both searchers share the game sidecar (sequential use in one thread)
        sn = DeterminizedDUCTChooser(
            model_new, tok, cfg, seed=wid, sidecar=sc)
        so = DeterminizedDUCTChooser(
            model_old, tok, cfg, seed=1000 + wid, sidecar=sc)
        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    g = jobs.pop()
                rng = random.Random(g)
                new_side = "p1" if g % 2 == 0 else "p2"
                old_side = "p2" if new_side == "p1" else "p1"
                sets = {s: team_sets[rng.choice(team_names)]
                        for s in ("p1", "p2")}
                bots = {
                    new_side: Bot(new_side, sets[new_side], sets[old_side],
                                  sn, usage, cfg),
                    old_side: Bot(old_side, sets[old_side], sets[new_side],
                                  so, usage, cfg)}
                winner, _ = run_game(sc, bots, sets, cfg, 0.0, rng)
                with lock:
                    score[0] += 1.0 if winner == new_side else \
                        0.5 if winner is None else 0.0
        finally:
            sc.close()
            sn.close()
            so.close()

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

def fork_model(src_ckpt, cfg, device):
    """BC checkpoint -> self-play starting point. v1 per-slot checkpoints are
    converted to the joint head (function-equivalent warm start)."""
    m = PolicyValueNet.load(src_ckpt, cfg, device)
    if m.policy_head == "slot":
        print(f"converting per-slot checkpoint {src_ckpt} to joint head")
        m = PolicyValueNet.from_slot(m, cfg)
    return m


def main(cfg=CFG):
    """Run resumable generate/train/gate iterations from config and CLI flags."""
    if len(sys.argv) > 1 and sys.argv[1] == "--_gen":
        _gen_subprocess(cfg)
        return
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    hours = float(opt("--hours", 0)) or None
    iters = int(opt("--iters", 0)) or None
    games = int(opt("--games", cfg.sp_games_per_iter))
    procs = int(opt("--procs", cfg.sp_procs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp_dir(cfg).mkdir(parents=True, exist_ok=True)
    buffer_dir(cfg).mkdir(parents=True, exist_ok=True)
    state_p = sp_dir(cfg) / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else \
        {"iter": -1, "history": []}

    last = sp_dir(cfg) / "sp_last.pt"
    if last.exists() and "--fresh" not in args:
        model = PolicyValueNet.load(last, cfg, device)
        print(f"resumed self-play at iteration {state['iter'] + 1}")
    else:
        stale = list(buffer_dir(cfg).glob("sp_*.npz"))
        assert not stale, \
            f"--fresh with {len(stale)} old buffer shards in {buffer_dir(cfg)}" \
            " — move or delete them first (iteration numbers would collide)"
        src = opt("--from", cfg.checkpoint_dir / "ckpt_best.pt")
        model = fork_model(src, cfg, device).to(device)
        model.save(last)
        state = {"iter": -1, "history": []}
        print(f"forked {src} -> {last}")
    tok = PositionTokenizer.load(cfg)
    assert model.hp["n_tokens"] == tok.n_tokens, \
        "checkpoint layout != current vocab.json — fork from a matching ckpt"
    assert model.policy_head == "joint"

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

        # generate: sp_procs subprocesses, each loading sp_last.pt
        per = [games // procs + (i < games % procs) for i in range(procs)]
        cmds = [subprocess.Popen(
            [sys.executable, __file__, "--_gen", str(it), str(i),
             str(per[i]), str(it * 131 + i)])
            for i in range(procs) if per[i] > 0]
        codes = [c.wait() for c in cmds]
        assert all(c == 0 for c in codes), f"generator failed: {codes}"

        n_new = sum(len(np.load(f)["value"])
                    for f in buffer_dir(cfg).glob(f"sp_{it:04d}_*.npz"))
        print(f"  generated {games} games / {n_new} samples "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)

        prev = PolicyValueNet.load(last, cfg, device)   # pre-training weights
        train_iteration(model, it, device, cfg)
        model.save(last)
        model.save(sp_dir(cfg) / f"sp_iter_{it:03d}.pt")

        g = None
        if "--no-gate" not in args and cfg.sp_gate_games > 0:
            g = gate(model, prev, tok, cfg, cfg.sp_gate_games)
            print(f"  gate vs prev iteration: {g:.0%} "
                  f"({cfg.sp_gate_games} games — coarse; run benchmark.py "
                  f"for a real series)", flush=True)

        state["iter"] = it
        state["history"].append({
            "iter": it, "games": games, "samples": n_new,
            "gate": g, "minutes": round((time.time() - t0) / 60, 1),
            "date": datetime.now().isoformat(timespec="minutes")})
        state_p.write_text(json.dumps(state, indent=1))
    print("\ndone — checkpoints in", sp_dir(cfg))


if __name__ == "__main__":
    main()
