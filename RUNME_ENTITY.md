# exp/entity-hybrid — run instructions (training box)

Lane: **pile-only** (§1 of EXPERIMENT_BRIEF.md). Branch-local edits touch
`agent_server.py` and `evaluation_common.py` (loader dispatch only); none of
the ten shared behavior files are modified.

**Idea:** balance transformer order-invariance against MLP positional wiring.
`models/entity_hybrid.py` encodes each Pokemon with a shared MLP (shared move
encoder inside), attends over 13 entities (global + 6 mine + 6 theirs, so
13² instead of 561² attention pairs), then flattens in fixed order into a
residual MLP trunk with the baseline's exact heads. Damage block dropped
(per the damage ablation). ~5.79M params, parameter-matched to the 1x
baseline. Target: best top-1/recall@6 on the fixed splits, then battle Elo.

## Prerequisites on the box

- `artifacts/vocab.json`, `usage_stats.json`, `dex.json`, `spreads.json`
  and the standard prepped shards under `artifacts/prepped/` (the same
  871,433/46,993/46,751 splits — this worktree ships them if you copied it
  whole; otherwise rsync from any machine that has run `data.py prep`).
- The pinned pokemon-showdown install (`requirements.txt` / `artifacts/node`)
  only for the battle phases, not for training.

## Run A — clean architecture comparison (protocol-matched to baseline)

```bash
python train_entity.py --smoke        # 30s self-test, must pass
python train_entity.py                # 8 epochs, same recipe as train.py
python evaluate.py artifacts/checkpoints/entity_ckpt_best.pt
```

Record static-mask top-1 / recall@6 / recall@16 / perplexity next to the
EXPERIMENTS.md capacity table (1x: 0.057 / 0.199 / 0.342 / 142.9;
10x: 0.078 / 0.266 / 0.427 / 102.7).

## Run B — permutation augmentation

```bash
rm artifacts/checkpoints/entity_ckpt_last.pt   # else it resumes run A
python train_entity.py --augment
python evaluate.py artifacts/checkpoints/entity_ckpt_best.pt
```

Augmentation remaps switch/move action labels and aux labels per row
(`entity_augment.py`); validation is never augmented, so val numbers stay
comparable. Keep whichever checkpoint evaluates better; rename it so run B
doesn't overwrite run A (`entity_ckpt_best.pt` -> `entity_A.pt` first).

## Battle phases (winner checkpoint)

```bash
python scenarios.py                                     # search gates
python benchmark.py play current baseline --quick 10    # NO — current loads ckpt_best;
# instead export and use the pile harness, which honors --ckpt:
python export_agent.py exp-entity-hybrid --agent search \
    --ckpt artifacts/checkpoints/entity_ckpt_best.pt \
    --architecture "EntityHybrid-5.8M" \
    --notes "entity-level transformer + MLP trunk, damage block dropped"
python round_robin.py play exp-entity-hybrid <anchor> --quick 10
```

`agent_server.py` on this branch dispatches on the checkpoint's recorded
architecture, so `--agent search --ckpt entity_ckpt_best.pt` runs the entity
model inside the standard DUCT search (and `--agent policy` for the
no-search bot).

## Gates before reporting done

```bash
python tests/test_documentation.py
python tests/test_agents.py
python scenarios.py
```

Then append results to `EXPERIMENTS.md` (what changed, static-mask metrics on
the fixed splits, which metric moved, quick-series result) and state where
the exported bundle lives.
