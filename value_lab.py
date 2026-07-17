"""Value-head experiment lab (exp/value-head): train alternative leaf-value
bricks on the frozen baseline, rank them on the held-out value dataset, and
package the winner as a playable combined checkpoint.

The baseline policy is never touched: every candidate replaces only the value
scalar (see models/value_heads.py). Selection happens on the validation split;
the test split is reported once, evaluate.py --value-style, so it stays an
honest holdout.

Candidates:
  control        the baseline's own linear+tanh value head (eval-only anchor)
  cls-mlp        frozen trunk, LayerNorm+MLP on the CLS state, CE loss
  attnpool       frozen trunk, learned-query attention pooling over all token
                 states + MLP, CE loss
  attnpool-mse   the attnpool architecture with the baseline's tanh+MSE loss
                 (ablation: does the win come from the loss or the pooling?)
  finetune       dedicated value net: baseline-initialized trunk fine-tuned
                 end-to-end on the value objective (2nd forward per leaf)

Data quality (all default-on; need ``python value_labels.py`` first — without
the sidecar the lab warns and trains/evaluates on the raw split):
  - abandoned games (<=1 total faint: rage-quit/disconnect, junk +-1 label)
    are dropped from training, val early-stopping, and eval;
  - over-long games (> --max-game-turns, default 14 — the stall/Trick-Room
    tail) are dropped the same way;
  - later positions are weighted a little more than early ones
    (--progression-floor, default 0.7 -> 1.0), since an early position's +-1
    label reflects many later turns of play it cannot see.
Metrics are reported on this clean subset; the full-set control/winner Brier
is also printed for comparability with EXPERIMENTS.md's baseline row.

CLI:
  python value_lab.py train  [--only A,B] [--ckpt PATH] [--quick] [--aux-w W]
                             [--progression-floor F] [--progression-gamma G]
                             [--max-game-turns N] [--keep-abandoned]
  python value_lab.py eval   [--ckpt PATH] [--quick] [--max-game-turns N]
                             [--keep-abandoned]
  python value_lab.py select [NAME] [--ckpt PATH] [--max-game-turns N]
                             [--keep-abandoned]
  python value_lab.py all    [--ckpt PATH] [--quick] [--aux-w W]
                             [--progression-floor F] [--progression-gamma G]
                             [--max-game-turns N] [--keep-abandoned]

Artifacts land under ``cfg.checkpoint_dir/value_lab/``. ``select`` writes
``combined_best.pt`` for ``agent_server.py --agent search-vh`` /
``export_agent.py --agent search-vh``.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("value_lab.py"):
        raise SystemExit(0)

import copy
import dataclasses
import json
import math
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

from agents.evaluation import BrickEvaluation, EvaluationStore
from config import CFG
from models.policy_value import PolicyValueNet
from models.value_heads import (AttnPoolHead, CLSMLPHead, ValueNet,
                                build_value_module, save_combined,
                                value_from_logit)
from tokenizer import N_MONS, PositionTokenizer
from train import Shards
from value_labels import load_margins, load_progression, load_sidecar

# name -> (kind, arch, loss/output, epochs, lr). Ordered cheap -> expensive.
CANDIDATES = {
    "cls-mlp":      {"kind": "head", "arch": "clsmlp",   "output": "bce",
                     "epochs": 6, "lr": 1e-3},
    "attnpool":     {"kind": "head", "arch": "attnpool", "output": "bce",
                     "epochs": 6, "lr": 1e-3},
    "attnpool-mse": {"kind": "head", "arch": "attnpool", "output": "mse",
                     "epochs": 6, "lr": 1e-3},
    "finetune":     {"kind": "net",  "arch": "attnpool", "output": "bce",
                     "epochs": 4, "lr": 5e-5},
}
PATIENCE = 2          # early-stop epochs without a val-Brier improvement
QUICK_ROWS = {"train": 2048, "val": 512, "test": 512}

# Data-quality defaults (all need `python value_labels.py` first; if the
# sidecar is absent they degrade to no-op with a warning). See value_labels.py.
DEFAULT_MAX_GAME_TURNS = 14    # drop rows from games lasting >14 turns (the
#                                long stall/Trick-Room tail, ~2% of games)
DEFAULT_DROP_ABANDONED = True  # drop rows from games that ended <=1 total
#                                faint (rage-quit/disconnect, junk z label)
# Mild "later positions matter a bit more" weighting, on by default: the
# earliest position is weighted 0.7x, ramping linearly to 1.0x at the last
# turn played. floor=1.0 disables it entirely.
DEFAULT_PROGRESSION_FLOOR = 0.7
DEFAULT_PROGRESSION_GAMMA = 1.0


def keep_mask(sidecar, n, max_game_turns=DEFAULT_MAX_GAME_TURNS,
              drop_abandoned=DEFAULT_DROP_ABANDONED):
    """Boolean ``[n]`` training/eval mask from the sidecar quality flags.

    Drops rows whose game ended in early abandonment (<=1 total faint) or ran
    longer than ``max_game_turns``. All-True (keep everything) when the
    sidecar is missing or both filters are disabled. ``[:n]`` keeps quick-mode
    row-subsets aligned to their truncated shard slice."""
    mask = np.ones(n, dtype=bool)
    if sidecar is None:
        return mask
    if drop_abandoned and "abandoned" in sidecar:
        mask &= sidecar["abandoned"][:n] == 0
    if max_game_turns and "final_turn" in sidecar:
        mask &= sidecar["final_turn"][:n] <= max_game_turns
    return mask


class ShardSlice:
    """--quick stand-in for ``train.Shards``: first shard only, truncated.

    Loading the full 871k-row split costs ~14GB and minutes of I/O — a smoke
    run must validate code paths, not throughput, so it reads one shard and
    keeps a few thousand rows."""

    def __init__(self, split, cfg=CFG):
        """Load the first ``<split>_*.npz`` shard, truncated to QUICK_ROWS."""
        files = sorted(glob(str(cfg.prepped_dir / f"{split}_*.npz")))
        if not files:
            raise FileNotFoundError(f"no {split} shards under {cfg.prepped_dir}")
        part, n = np.load(files[0]), QUICK_ROWS[split]
        self.tokens = part["tokens"][:n]
        self.value = part["value"][:n]
        self.weight = part["weight"][:n].astype(np.float32)
        self.weight = self.weight / max(float(self.weight.mean()), 1e-9)

    def __len__(self):
        """Return the number of retained rows."""
        return len(self.tokens)


def load_split(split, cfg=CFG, quick=False):
    """Return the full ``Shards`` split, or a tiny ``ShardSlice`` in quick mode."""
    return ShardSlice(split, cfg) if quick else Shards(split, cfg)


class SplitView:
    """Boolean-masked row view of a split (tokens/value/weight only).

    Used to hold out the abandoned and over-long games from validation and
    test so selection and reporting target the positions the value head is
    actually meant to judge."""

    def __init__(self, ds, mask):
        """Slice ``ds``'s fields to the rows where ``mask`` is true."""
        self.tokens = ds.tokens[mask]
        self.value = ds.value[mask]
        self.weight = ds.weight[mask]

    def __len__(self):
        """Return the number of retained rows."""
        return len(self.tokens)


def out_dir(cfg=CFG):
    """Return (and create) the lab's checkpoint directory."""
    d = cfg.checkpoint_dir / "value_lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_module(spec, base):
    """Construct one candidate's trainable module on the base's device."""
    d = int(base.hp["model_cfg"]["d_model"])
    if spec["kind"] == "net":
        module = ValueNet.from_base(base)
    elif spec["arch"] == "clsmlp":
        module = CLSMLPHead(d)
    else:
        module = AttnPoolHead(d)
    return module.to(next(base.parameters()).device)


def _autocast(device):
    """bf16 autocast on cuda, no-op elsewhere."""
    return torch.autocast(device, dtype=torch.bfloat16,
                          enabled=device == "cuda")


def _forward(base, module, spec, tokens):
    """One candidate forward: trunk states for heads, raw tokens for nets."""
    if spec["kind"] == "head":
        with torch.no_grad():
            h = base.encoder(base.emb(tokens) + base.pos)
        return module(h)
    return module(tokens)


def progression_weight(p, floor=1.0, gamma=1.0):
    """Map a [0,1] game-progression fraction to a loss-weight multiplier.

    ``floor`` is the minimum multiplier (at team preview, p=0); it ramps to
    1.0 at p=1 (the last turn played) following ``p ** gamma``. floor=1.0 is
    the identity (no reweighting) — the default, so existing behavior is
    unchanged unless explicitly opted into via --progression-floor."""
    return floor + (1.0 - floor) * np.clip(p, 0.0, 1.0) ** gamma


def train_candidate(name, spec, base, tr, va, margins_tr, device, cfg,
                    aux_w=0.25, quick=False, progression_tr=None,
                    progression_floor=1.0, progression_gamma=1.0,
                    train_idx=None):
    """Train one candidate with early stopping; save and return its record.

    Loss = sample-weighted CE (or MSE for the ablation) on the +-1 outcome
    plus ``aux_w`` x sample-weighted MSE on the sidecar margins when built.
    ``train_idx`` (if given) restricts training to those row indices — the
    data-quality filter drops early-abandoned and over-long games upstream.
    When ``progression_tr`` is given and ``progression_floor < 1.0``, the
    per-row sample weight is additionally multiplied by
    ``progression_weight`` — a proxy for "how much of the eventual z this
    position could plausibly explain" (see value_labels.py), so a turn-2
    row that got labeled by 20 more turns of play (opponent blunders,
    comebacks) is fit less aggressively than a turn-20 row close to the
    actual result. Early stopping tracks unweighted validation Brier on the
    (already-filtered) ``va`` — the selection metric — so training and
    selection cannot disagree about what "better" means."""
    torch.manual_seed(0)
    module = build_module(spec, base)
    n_params = sum(p.numel() for p in module.parameters())
    use_aux = margins_tr is not None
    use_progression = progression_tr is not None and progression_floor < 1.0
    pool = np.arange(len(tr)) if train_idx is None else np.asarray(train_idx)
    print(f"\n=== {name}: {spec} | {n_params / 1e6:.2f}M trainable params | "
          f"{len(pool)}/{len(tr)} rows after quality filter | "
          f"aux margins {'on' if use_aux else 'off (run value_labels.py)'} | "
          + (f"progression weight floor={progression_floor:.2f} "
             f"gamma={progression_gamma:.2f}" if use_progression
             else "progression weight off") + " ===")
    opt = torch.optim.AdamW(module.parameters(), lr=spec["lr"],
                            weight_decay=0.01)
    epochs = 1 if quick else spec["epochs"]
    bs = cfg.batch_size
    steps_per_epoch = math.ceil(len(pool) / bs)
    total = max(1, steps_per_epoch * epochs)
    warmup = min(100, total // 10 + 1)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(0)
    best = {"brier": float("inf"), "state": None, "epoch": -1}
    history = []
    for epoch in range(epochs):
        module.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        order = rng.permutation(pool)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            t = torch.as_tensor(tr.tokens[idx].astype(np.int64), device=device)
            z = torch.as_tensor(tr.value[idx].astype(np.float32), device=device)
            sw = tr.weight[idx].astype(np.float32)
            if use_progression:
                sw = sw * progression_weight(progression_tr[idx],
                                             progression_floor,
                                             progression_gamma)
            w = torch.as_tensor(sw, device=device)
            with _autocast(device):
                out = _forward(base, module, spec, t)
            logit = out[:, 0].float()
            if spec["output"] == "bce":
                per = F.binary_cross_entropy_with_logits(
                    logit, (z + 1) / 2, reduction="none")
            else:
                per = (torch.tanh(logit) - z) ** 2
            loss = (per * w).mean()
            if use_aux:
                m = torch.as_tensor(margins_tr[idx], device=device,
                                    dtype=torch.float32)
                loss = loss + aux_w * (
                    ((out[:, 1:].float() - m) ** 2).mean(-1) * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            run_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
        val_v = candidate_values(base, module, spec, va.tokens, device, cfg)
        brier = float(np.mean((((val_v + 1) / 2)
                               - ((va.value.astype(np.float64) + 1) / 2)) ** 2))
        history.append({"epoch": epoch, "train_loss": run_loss / max(1, seen),
                        "val_brier": brier})
        print(f"epoch {epoch:2d} | train loss {run_loss / max(1, seen):.4f} | "
              f"val Brier {brier:.4f} | {time.time() - t0:.0f}s")
        if brier < best["brier"]:
            best = {"brier": brier, "epoch": epoch,
                    "state": copy.deepcopy(
                        {k: v.cpu() for k, v in module.state_dict().items()})}
        elif epoch - best["epoch"] >= PATIENCE:
            print(f"early stop (no val improvement for {PATIENCE} epochs)")
            break
    module.load_state_dict(best["state"])
    record = {"value": {"kind": spec["kind"], "output": spec["output"],
                        "hp": module.hp, "state": best["state"]},
              "spec": {k: v for k, v in spec.items()},
              "history": history, "best_val_brier": best["brier"],
              "aux_margins": use_aux, "aux_w": aux_w,
              "train_rows": int(len(pool)),
              "progression": {"on": use_progression,
                              "floor": progression_floor,
                              "gamma": progression_gamma}}
    torch.save(record, out_dir(cfg) / f"{name}.pt")
    print(f"saved {out_dir(cfg) / (name + '.pt')} "
          f"(best val Brier {best['brier']:.4f} @ epoch {best['epoch']})")
    return record


@torch.no_grad()
def candidate_logits(base, module, spec, tokens, device, cfg):
    """Batched win-logits for one candidate over a token matrix."""
    module.eval()
    outs = []
    for s in range(0, len(tokens), cfg.batch_size):
        t = torch.as_tensor(tokens[s:s + cfg.batch_size].astype(np.int64),
                            device=device)
        with _autocast(device):
            out = _forward(base, module, spec, t)
        outs.append(out[:, 0].float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def candidate_values(base, module, spec, tokens, device, cfg, temperature=1.0):
    """Batched values in [-1, 1] for one candidate over a token matrix."""
    logits = candidate_logits(base, module, spec, tokens, device, cfg)
    return value_from_logit(torch.as_tensor(logits), spec["output"],
                            temperature).numpy()


@torch.no_grad()
def control_values(base, tokens, cfg):
    """The baseline's own value head over a token matrix (the anchor row)."""
    vals = []
    for s in range(0, len(tokens), cfg.batch_size):
        _, v, _ = base.predict_batch(tokens[s:s + cfg.batch_size])
        vals.append(v)
    return np.concatenate(vals).astype(np.float64)


def auc_score(y, p):
    """Rank-based (Mann-Whitney) AUC with average ranks on ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def value_metrics(v, z):
    """evaluate.py --value's headline numbers plus AUC, as one dict."""
    v = np.clip(np.asarray(v, dtype=np.float64), -1.0, 1.0)
    z = np.asarray(z, dtype=np.float64)
    p, y = (v + 1) / 2, (z + 1) / 2
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p > lo) & (p <= hi) if lo > 0 else (p >= 0) & (p <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return {"brier": float(np.mean((p - y) ** 2)),
            "mse": float(np.mean((v - z) ** 2)),
            "mae": float(np.mean(np.abs(v - z))),
            "sign_acc": float(np.mean(np.sign(v) == np.sign(z))),
            "auc": auc_score(y, p),
            "ece": float(ece),
            "mean_abs_v": float(np.mean(np.abs(v)))}


def _token_lookup(tok, prefix, cast=float):
    """Id-indexed array of the numeric suffix of ``prefix`` tokens (else NaN)."""
    out = np.full(max(tok.vocab.values()) + 1, np.nan)
    for name, i in tok.vocab.items():
        if name.startswith(prefix):
            out[i] = cast(name[len(prefix):])
    return out


def floors(tokens, z, tok):
    """The two sign-accuracy floors from evaluate.py --value."""
    y = (np.asarray(z, dtype=np.float64) + 1) / 2
    hp = _token_lookup(tok, "HP_")
    hp_of = lambda base_i: np.nansum(
        hp[tokens[:, [base_i + k * tok.mon_block + 4
                      for k in range(N_MONS)]]], axis=1)
    diff = hp_of(tok.my_base) - hp_of(tok.opp_base)
    edge = diff != 0
    hp_acc = float(np.mean(np.sign(diff[edge]) == np.sign(z[edge]))) \
        if edge.any() else float("nan")
    return {"always_win": float(max(y.mean(), 1 - y.mean())),
            "hp_differential": hp_acc}


def phase_table(v, z, tokens, tok):
    """Per turn-bucket ``(n, sign_acc, brier, mean|v|)`` rows."""
    v = np.clip(np.asarray(v, dtype=np.float64), -1, 1)
    z = np.asarray(z, dtype=np.float64)
    turn = _token_lookup(tok, "TURN_", cast=int)
    buckets = turn[tokens[:, 1]]
    rows = []
    for b in sorted(set(buckets[~np.isnan(buckets)])):
        m = buckets == b
        rows.append((int(b), int(m.sum()),
                     float(np.mean(np.sign(v[m]) == np.sign(z[m]))),
                     float(np.mean((((v[m] + 1) / 2) - ((z[m] + 1) / 2)) ** 2)),
                     float(np.mean(np.abs(v[m])))))
    return rows


def trained_candidates(cfg=CFG):
    """Return ``{name: saved record}`` for every trained candidate on disk."""
    found = {}
    for name in CANDIDATES:
        path = out_dir(cfg) / f"{name}.pt"
        if path.exists():
            found[name] = torch.load(path, map_location="cpu",
                                     weights_only=False)
    return found


def _restore(record, base, device):
    """Rebuild a saved candidate module on ``device``."""
    module = build_value_module(record["value"], base).to(device)
    module.load_state_dict(record["value"]["state"])
    return module


def evaluate_all(base, device, cfg, quick=False,
                 max_game_turns=DEFAULT_MAX_GAME_TURNS,
                 drop_abandoned=DEFAULT_DROP_ABANDONED):
    """Rank control + every trained candidate on val, report test; persist.

    Metrics are computed on the *clean* subset — abandoned and over-long
    games held out (see keep_mask) — so selection targets the positions the
    value head is meant to judge; the full-set control/winner Brier is printed
    as a footnote for comparability with EXPERIMENTS.md's baseline row. Prints
    the selection table (val Brier is the criterion), the winner's phase
    breakdown, appends one ``leaf_evaluator`` BrickEvaluation per row, and
    writes ``results.json`` for ``select``."""
    tok = PositionTokenizer.load(cfg)
    va_full, te_full = load_split("val", cfg, quick), load_split("test", cfg, quick)
    va_side, te_side = load_sidecar("val", cfg), load_sidecar("test", cfg)
    va_mask = keep_mask(va_side, len(va_full), max_game_turns, drop_abandoned)
    te_mask = keep_mask(te_side, len(te_full), max_game_turns, drop_abandoned)
    va, te = SplitView(va_full, va_mask), SplitView(te_full, te_mask)
    if te_side is None:
        print("NOTE: no artifacts/value_labels/ sidecar — evaluating on the "
              "FULL split (run 'python value_labels.py' to hold out abandoned "
              "and over-long games)")
    else:
        print(f"clean-subset eval: kept val {va_mask.sum()}/{len(va_full)}, "
              f"test {te_mask.sum()}/{len(te_full)} rows "
              f"(dropped abandoned + games >{max_game_turns} turns)")

    rows, store = {}, EvaluationStore(cfg=cfg)
    v_control_te = control_values(base, te.tokens, cfg)
    rows["control"] = {"val": value_metrics(control_values(base, va.tokens, cfg),
                                            va.value),
                       "test": value_metrics(v_control_te, te.value)}
    per_split_values = {"control": v_control_te}
    for name, record in trained_candidates(cfg).items():
        spec = record["spec"]
        module = _restore(record, base, device)
        v_val = candidate_values(base, module, spec, va.tokens, device, cfg)
        v_te = candidate_values(base, module, spec, te.tokens, device, cfg)
        rows[name] = {"val": value_metrics(v_val, va.value),
                      "test": value_metrics(v_te, te.value)}
        per_split_values[name] = v_te
    fl = floors(te.tokens, te.value, tok)

    cols = ("brier", "mse", "sign_acc", "auc", "ece", "mean_abs_v")
    for split in ("val", "test"):
        print(f"\n=== {split} split ===")
        print(f"{'':14s}" + "".join(f"{c:>11s}" for c in cols))
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["val"]["brier"]):
            print(f"{name:14s}"
                  + "".join(f"{r[split][c]:11.4f}" for c in cols))
    print(f"\ntest sign-accuracy floors: always-win {fl['always_win']:.1%}, "
          f"HP differential {fl['hp_differential']:.1%}")

    ranked = sorted((n for n in rows if n != "control"),
                    key=lambda n: rows[n]["val"]["brier"])
    if ranked:
        best = ranked[0]
        print(f"\nselection (val Brier): {best}")
        if rows[best]["val"]["brier"] >= rows["control"]["val"]["brier"]:
            print("WARNING: no candidate beats the control on val Brier — "
                  "the honest result is 'keep the baseline head'")
        print(f"\n{best} by game phase (clean test subset):")
        print(f"    {'turn bucket':>12s} {'n':>7s} {'sign acc':>9s} "
              f"{'Brier':>7s} {'mean |v|':>9s}")
        for b, n, acc, brier, mabs in phase_table(per_split_values[best],
                                                  te.value, te.tokens, tok):
            print(f"    {b:>12d} {n:7d} {acc:9.1%} {brier:7.4f} {mabs:9.3f}")
        if te_side is not None and len(te_full) > len(te):
            c_full = value_metrics(control_values(base, te_full.tokens, cfg),
                                   te_full.value)["brier"]
            b_mod = _restore(trained_candidates(cfg)[best], base, device)
            b_full = value_metrics(
                candidate_values(base, b_mod,
                                 trained_candidates(cfg)[best]["spec"],
                                 te_full.tokens, device, cfg),
                te_full.value)["brier"]
            print(f"\nfull test set incl. abandoned + long games (comparable "
                  f"to EXPERIMENTS.md baseline): control Brier {c_full:.4f}, "
                  f"{best} Brier {b_full:.4f}")

    filter_cfg = {"max_game_turns": max_game_turns,
                  "drop_abandoned": drop_abandoned,
                  "clean_test_rows": int(len(te)),
                  "full_test_rows": int(len(te_full))}
    for name, r in rows.items():
        store.append(BrickEvaluation(
            brick_impl=f"exp-value-head/{name}", suite="leaf_evaluator",
            metrics={f"{s}_{k}": v for s in ("val", "test")
                     for k, v in r[s].items()},
            cases=len(te.value), config={},
            metadata={"floors": fl, "filter": filter_cfg}))
    (out_dir(cfg) / "results.json").write_text(json.dumps(
        {"rows": rows, "floors": fl, "ranking": ranked,
         "filter": filter_cfg}, indent=1))
    print(f"\nwrote {out_dir(cfg) / 'results.json'} and appended "
          f"brick_evaluations rows")

    print("\n--- paste-ready EXPERIMENTS.md table (clean test subset) ---")
    print("| value brick | test Brier | test sign acc | test AUC | test ECE |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name in ["control"] + ranked:
        r = rows[name]["test"]
        print(f"| {name} | {r['brier']:.4f} | {r['sign_acc']:.3f} | "
              f"{r['auc']:.3f} | {r['ece']:.3f} |")
    return rows, ranked


def fit_temperature(logits, y, output="bce"):
    """Grid-fit the post-hoc calibration temperature on validation logits."""
    best_t, best_loss = 1.0, float("inf")
    for t in np.logspace(-1.2, 1.2, 49):
        scaled = torch.as_tensor(logits / t)
        if output == "bce":
            p = torch.sigmoid(scaled).numpy()
            loss = float(-np.mean(y * np.log(np.clip(p, 1e-12, 1))
                                  + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        else:
            p = (torch.tanh(scaled).numpy() + 1) / 2
            loss = float(np.mean((p - y) ** 2))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def select(base, base_ckpt_raw, device, cfg, name=None, quick=False,
           max_game_turns=DEFAULT_MAX_GAME_TURNS,
           drop_abandoned=DEFAULT_DROP_ABANDONED):
    """Package one candidate (default: best val Brier) as a combined agent.

    The calibration temperature is fitted on the same clean validation subset
    used for selection, so the deployed probabilities are calibrated on the
    positions the head is meant to judge."""
    results = json.loads((out_dir(cfg) / "results.json").read_text())
    ranking = results["ranking"]
    assert ranking, "no trained candidates — run 'value_lab.py train' first"
    name = name or ranking[0]
    record = trained_candidates(cfg)[name]
    spec = record["spec"]
    module = _restore(record, base, device)
    va_full = load_split("val", cfg, quick)
    va = SplitView(va_full, keep_mask(load_sidecar("val", cfg), len(va_full),
                                      max_game_turns, drop_abandoned))
    logits = candidate_logits(base, module, spec, va.tokens, device, cfg)
    temperature = fit_temperature(
        logits, (va.value.astype(np.float64) + 1) / 2, spec["output"])
    v_cal = value_from_logit(torch.as_tensor(logits), spec["output"],
                             temperature).numpy()
    print(f"{name}: fitted temperature {temperature:.3f} "
          f"(val Brier {value_metrics(v_cal, va.value)['brier']:.4f})")
    meta = {"candidate": name, "spec": {k: v for k, v in spec.items()},
            "metrics": results["rows"].get(name), "temperature": temperature}
    module_cpu = module.to("cpu")
    for fname in (f"combined_{name}.pt", "combined_best.pt"):
        save_combined(out_dir(cfg) / fname, base_ckpt_raw, module_cpu,
                      spec["kind"], spec["output"], temperature, meta)
        print(f"wrote {out_dir(cfg) / fname}")
    print("\nnext:\n"
          f"  python export_agent.py exp-value-head --agent search-vh "
          f"--ckpt {out_dir(cfg) / 'combined_best.pt'} "
          f"--architecture 'DUCT+ValueSwap({name})' "
          f"--notes 'baseline policy, {name} value brick'\n"
          "  python round_robin.py play exp-value-head <anchor> --quick 10")


def main(cfg=CFG):
    """Dispatch train/eval/select/all from argv."""
    args = sys.argv[1:]
    cmd = args[0] if args and not args[0].startswith("--") else "all"

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    quick = "--quick" in args
    if quick:
        cfg = dataclasses.replace(cfg, batch_size=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    ckpt = opt("--ckpt", cfg.checkpoint_dir / "ckpt_best.pt")
    base = PolicyValueNet.load(ckpt, cfg, device)
    base.eval()
    base_raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"base checkpoint: {ckpt} | device {device}")

    # Data-quality filter (applies to train, val early-stopping, and eval).
    max_game_turns = int(opt("--max-game-turns", DEFAULT_MAX_GAME_TURNS))
    drop_abandoned = "--keep-abandoned" not in args and DEFAULT_DROP_ABANDONED

    if cmd in ("train", "all"):
        only = opt("--only")
        names = [n.strip() for n in only.split(",")] if only \
            else list(CANDIDATES)
        aux_w = float(opt("--aux-w", 0.25))
        prog_floor = float(opt("--progression-floor", DEFAULT_PROGRESSION_FLOOR))
        prog_gamma = float(opt("--progression-gamma", DEFAULT_PROGRESSION_GAMMA))
        tr = load_split("train", cfg, quick)
        va_full = load_split("val", cfg, quick)
        side_tr = load_sidecar("train", cfg)
        side_va = load_sidecar("val", cfg)
        margins_tr = None if side_tr is None else side_tr["margins"]
        progression_tr = None if side_tr is None else side_tr["progression"]
        if side_tr is None:
            print("WARNING: no artifacts/value_labels/ sidecar — training on "
                  "the RAW split (no abandoned/long-game filter, no margin "
                  "aux, no progression weighting). Run 'python "
                  "value_labels.py' to enable them.")
        if quick and side_tr is not None:
            margins_tr = margins_tr[:len(tr.tokens)]
            progression_tr = progression_tr[:len(tr.tokens)]
        train_idx = np.where(
            keep_mask(side_tr, len(tr.tokens), max_game_turns,
                      drop_abandoned))[0]
        va = SplitView(va_full, keep_mask(side_va, len(va_full),
                                          max_game_turns, drop_abandoned))
        print(f"train {len(train_idx)}/{len(tr.tokens)} (filtered) / "
              f"val {len(va)}/{len(va_full)} transitions")
        for name in names:
            train_candidate(name, CANDIDATES[name], base, tr, va, margins_tr,
                            device, cfg, aux_w=aux_w, quick=quick,
                            progression_tr=progression_tr,
                            progression_floor=prog_floor,
                            progression_gamma=prog_gamma,
                            train_idx=train_idx)
    if cmd in ("eval", "all"):
        evaluate_all(base, device, cfg, quick=quick,
                     max_game_turns=max_game_turns,
                     drop_abandoned=drop_abandoned)
    if cmd in ("select", "all"):
        picked = args[1] if cmd == "select" and len(args) > 1 \
            and not args[1].startswith("--") else None
        select(base, base_raw, device, cfg, name=picked, quick=quick,
               max_game_turns=max_game_turns, drop_abandoned=drop_abandoned)


if __name__ == "__main__":
    main()
