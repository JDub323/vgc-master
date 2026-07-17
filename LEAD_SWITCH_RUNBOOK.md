# exp/lead-switch — runbook

Everything below assumes the big box (or any machine with `ckpt_best.pt`,
the phase-1 artifacts, and Node). On the Windows laptop, run through WSL:
`wsl` + `.venv/bin/python`. The worktree lives at `../vgc-bot-lead-switch`
(branch `exp/lead-switch`); the pile is the sibling `../vgc-pile`.

## 0. One-time sanity (any box)

```bash
python tests/test_lead_switch.py      # selector unit gates (no Node/ckpt needed)
python tests/test_documentation.py
python tests/test_agents.py
python scenarios.py
```

## 1. Train LeadNet (cheap — laptop or big box)

Needs `artifacts/parsed/*.pkl` (data.py parse) and `artifacts/vocab.json`.

```bash
python train_leads.py                          # ~1M params, minutes on CPU
# smoke: python train_leads.py --max-battles 500 --epochs 2
```

Writes `artifacts/checkpoints/leadnet.pt` and prints lead-pair top-1/top-3
vs. the human choice plus the "humans led with the first two only X% of
games" floor — record those in EXPERIMENTS.md.

## 2. Export the contestants (big box, needs ckpt_best.pt)

```bash
# the anchor: plain baseline behavior (team 1234, random forced switch)
python export_agent.py rr-baseline --agent search --notes "frozen 1x anchor"

# the three experiment variants (same frozen model, better positioning only)
python export_lead_switch.py exp-ls-expert --variant expert \
    --notes "expert leads + expert switch-ins"
python export_lead_switch.py exp-ls-value --variant value \
    --notes "frozen value head leads + switch-ins"
python export_lead_switch.py exp-ls-nn --variant nn \
    --notes "LeadNet preview + value switch-ins"

python round_robin.py list
```

Bundles are immutable — re-export under a new name after any code change.
If training and the tournament happen on different machines, `rsync -a`
the bundle directories into the tournament box's `../vgc-pile`.

## 3. Triage, then the real series

```bash
# 10-game triage per variant ("not obviously broken", NOT a verdict)
python round_robin.py play exp-ls-expert rr-baseline --quick 10
python round_robin.py play exp-ls-value  rr-baseline --quick 10
python round_robin.py play exp-ls-nn     rr-baseline --quick 10

# full 100-game series (±10% Wilson at 100 games) — the actual result
python round_robin.py play exp-ls-expert rr-baseline
python round_robin.py play exp-ls-value  rr-baseline
python round_robin.py play exp-ls-nn     rr-baseline
python round_robin.py standings
```

Watch the live dashboard on port 8020 (printed URL); `--no-live` for
headless. `--workers 4` speeds a series up on a big box.

## 4. Ablations worth one series each (optional)

The server splits the two decisions, so you can isolate them:

```bash
# switch-ins only (preview stays team-1234)
python export_agent.py exp-ls-swonly --agent search --entrypoint \
  "python lead_switch_server.py --agent search --ckpt artifacts/checkpoints/ckpt.pt --leads first4 --switch expert" \
  --architecture "DUCT+expert-switch-only"

# leads only (switches stay random)
python export_agent.py exp-ls-leadonly --agent search --entrypoint \
  "python lead_switch_server.py --agent search --ckpt artifacts/checkpoints/ckpt.pt --leads expert --switch random" \
  --architecture "DUCT+expert-leads-only"
```

## 5. Record

Append results to the `Team preview + forced switch-ins` section of
EXPERIMENTS.md: variant, series score, Wilson CI, Elo, and which selector
(leads vs switches) carried the change. Negative results too.
