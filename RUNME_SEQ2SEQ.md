# exp/seq2seq-pointer — run instructions (training box)

Lane: **pile-only** (§1 of EXPERIMENT_BRIEF.md). Branch-local edits touch
`agent_server.py` and `evaluation_common.py` (loader dispatch only); none of
the ten shared behavior files are modified.

**Idea:** stop scoring 1521 fixed joint outputs and point at the legal moves
instead. `models/seq2seq.py` reuses the entity-hybrid encoder (13 entity
vectors, damage block dropped) and feeds one transformer decoder layer whose
*inputs are the legal actions*: 39 candidate tokens per slot built from
action-index + content (the acting mon's move token / the switch target's
species token) + target-code + mega embeddings, illegal candidates masked.
Slot A is a 39-way pointer head; slot B is a `[39, 39]` pairwise head
conditioned on A (chain rule), masked by legal-B ∧ static `joint_ok`.
`predict_batch` expands `P(a)·P(b|a)` onto the flat `[1521]` grid, so DUCT
search, `evaluate.py`, and `agent_server.py` run unchanged. ~5.8M params,
parameter-matched to the 1x baseline and the entity hybrid.

Legality is reconstructed from tokens via
`evaluation_common.PositionLegality` (permissive superset — disabled moves,
PP, trapping, bring-four are unknowable from tokens), the **same source at
train and play time**. Labels are trained as set-CE over the target-projected
label set (KNOWN_ISSUES.md #3: 8.7% of raw labels carry a target code the
legal set never contains).

## Prerequisites on the box

- `artifacts/vocab.json`, `usage_stats.json`, `dex.json`, `spreads.json`
  and the standard prepped shards under `artifacts/prepped/` (the same
  871,433/46,993/46,751 splits — rsync from any machine that has run
  `data.py prep`).
- The pinned pokemon-showdown install (`requirements.txt` / `artifacts/node`)
  only for the battle phases, not for training.

## Run order

```bash
python train_seq2seq.py --smoke      # ~1 min self-test, must pass; prints
                                     # param count and the B=2 predict_batch
                                     # latency incl. legality reconstruction
python seq2seq_prep.py               # one-time sidecar build, ~30-60 min of
                                     # per-row Python; prints projected-label
                                     # and label-outside-legal-superset rates
python train_seq2seq.py              # 8 epochs, train.py's exact recipe
python evaluate.py artifacts/checkpoints/seq2seq_ckpt_best.pt
```

**Quoting metrics:** use only the **position-legal table** from
`evaluate.py` (and `train_seq2seq.py`'s `top1_legal`). The static-mask table
is meaningless for this model — it masks legality *by construction*, so
static-mask numbers are inflated and not comparable to the historical
EXPERIMENTS.md rows. Say so explicitly when reporting.

## Battle phases (best checkpoint)

```bash
python scenarios.py                                     # search gates
python export_agent.py exp-seq2seq-pointer --agent search \
    --ckpt artifacts/checkpoints/seq2seq_ckpt_best.pt \
    --architecture "Seq2SeqPointer-5.8M" \
    --notes "legal-move pointer decoder over the entity encoder, set-CE on projected labels"
python round_robin.py play exp-seq2seq-pointer rr-baseline --quick 10
# same-encoder comparison — isolates the pointer head vs the joint head:
python round_robin.py play exp-seq2seq-pointer exp-entity-hybrid --quick 10
```

`agent_server.py` on this branch dispatches on the checkpoint's recorded
architecture, so `--agent search --ckpt seq2seq_ckpt_best.pt` runs the
pointer model inside the standard DUCT search (`--agent policy` for the
no-search bot). Watch the seconds/move column: `predict_batch` runs
per-row Python legality reconstruction on every call (the `--smoke` B=2
timing is the per-expansion overhead); a search-time regression is a
finding, not something to hide.

## Gates before reporting done

```bash
python tests/test_documentation.py
python tests/test_agents.py
python scenarios.py
```

Then append results to `EXPERIMENTS.md` — what changed, **position-legal**
metrics on the fixed splits (state that static-mask numbers are structurally
incomparable for this model), the quick-series results vs `rr-baseline` and
vs `exp-entity-hybrid`, and where the exported bundle lives. Never push —
the user pushes.
