"""LeadNet: a small trained network for team preview (bring 4, lead 2).

Team preview is the one decision the replay dataset labels almost perfectly:
the two leads are public in every game's first turn, the brought four are the
mons that appeared (partially observed when a game ends before all four show).
LeadNet imitates strong human preview play from those labels, weighted by the
same rating/format/recency scheme the base model trained on, with the losing
side's picks down-weighted so the target skews toward choices that won.

Architecture mirrors what already worked for the base model at a fraction of
the size: shared token embeddings (ids come from the SAME vocab.json the
baseline tokenizer uses, so species/item/move namespaces stay aligned) -> a
2-layer transformer over 13 tokens (CLS + my 6 mons + their 6 species; a mon
token is the sum of its species/item/ability/mean-move embeddings plus a side
embedding) -> two heads:

  bring head  per-my-mon logit "this one is in the four"
  pair head   one logit per lead pair (i < j), an MLP over
              [h_i + h_j, h_i * h_j]

Inference enumerates the 90 legal bring/lead combos and adds the pair logit
to a weighted mean of the four bring logits. Everything here is new,
experiment-local capacity; the frozen baseline net is untouched.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.lead_switch.expert import preview_order
from agents.lead_switch.lscfg import LSCFG
from data import battle_weight, iter_battles, sid

N_MONS = 6
PAIRS = list(combinations(range(N_MONS), 2))          # 15, fixed order
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}
MON_FEATS = 7                                          # species item abil 4 moves


class LeadNet(nn.Module):
    """Tiny transformer over both preview rosters with bring/pair heads."""

    def __init__(self, vocab_size, ls=LSCFG):
        """Build embeddings, encoder, and the two heads from ``LSConfig``."""
        super().__init__()
        d = ls.nn_d_model
        self.hp = {"vocab_size": vocab_size,
                   "d_model": d, "n_layers": ls.nn_layers,
                   "n_heads": ls.nn_heads, "dropout": ls.nn_dropout}
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.side = nn.Embedding(3, d)                 # CLS / mine / theirs
        layer = nn.TransformerEncoderLayer(
            d, ls.nn_heads, 4 * d, ls.nn_dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, ls.nn_layers)
        self.norm = nn.LayerNorm(d)
        self.bring_head = nn.Linear(d, 1)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, my_feats, opp_species):
        """my_feats [B,6,7] ids, opp_species [B,6] ids ->
        (bring logits [B,6], pair logits [B,15])."""
        B = my_feats.shape[0]
        mine = (self.emb(my_feats[:, :, 0])
                + self.emb(my_feats[:, :, 1])
                + self.emb(my_feats[:, :, 2])
                + self.emb(my_feats[:, :, 3:]).mean(dim=2)
                + self.side.weight[1])
        theirs = self.emb(opp_species) + self.side.weight[2]
        cls = self.side.weight[0].expand(B, 1, -1)
        h = self.encoder(torch.cat([cls, mine, theirs], dim=1))
        h = self.norm(h)
        hm = h[:, 1:1 + N_MONS]                        # my mon outputs
        bring = self.bring_head(hm).squeeze(-1)
        pf = torch.stack([torch.cat([hm[:, i] + hm[:, j], hm[:, i] * hm[:, j]],
                                    dim=-1) for i, j in PAIRS], dim=1)
        pair = self.pair_mlp(pf).squeeze(-1)
        return bring, pair

    def save(self, path):
        """Write hyperparameters and weights; return ``None``."""
        torch.save({"hp": self.hp, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, ls=LSCFG, device="cpu"):
        """Return a checkpoint-restored LeadNet on ``device``."""
        ck = torch.load(path, map_location=device, weights_only=False)
        import dataclasses
        ls = dataclasses.replace(ls, nn_d_model=ck["hp"]["d_model"],
                                 nn_layers=ck["hp"]["n_layers"],
                                 nn_heads=ck["hp"]["n_heads"],
                                 nn_dropout=ck["hp"]["dropout"])
        m = cls(ck["hp"]["vocab_size"], ls).to(device)
        m.load_state_dict(ck["state"])
        return m


# ---------------------------------------------------------------------------
# dataset extraction from parsed battles
# ---------------------------------------------------------------------------

def _tok_ids(tok, set_):
    """One mon's [species, item, ability, 4 moves] vocab-id feature row."""
    unk = tok.vocab["UNK"]
    moves = [tok.vocab.get(f"move:{m}", unk) for m in set_["moves"][:4]]
    moves += [0] * (4 - len(moves))
    return ([tok.vocab.get(f"species:{sid(set_['species'])}", unk),
             tok.vocab.get(f"item:{set_['item']}", unk) if set_["item"] else 0,
             tok.vocab.get(f"ability:{set_['ability']}", unk)
             if set_["ability"] else 0] + moves)


def team_features(tok, my_sets, opp_species_sids):
    """(my_feats [6,7], opp_species [6]) int arrays for one preview."""
    unk = tok.vocab["UNK"]
    my = [_tok_ids(tok, s) for s in my_sets[:N_MONS]]
    my += [[0] * MON_FEATS] * (N_MONS - len(my))
    opp = [tok.vocab.get(f"species:{s}", unk) for s in opp_species_sids[:N_MONS]]
    opp += [0] * (N_MONS - len(opp))
    return (np.array(my, dtype=np.int64), np.array(opp, dtype=np.int64))


def extract_examples(files, tok, cfg, ls=LSCFG, max_battles=None):
    """Stream parsed battles into preview training examples.

    Returns {"train"|"val"|"test": list of example dicts} where an example
    holds my_feats/opp_species id arrays, the human lead-pair index, bring
    targets with a mask (appeared mons are positive; negatives only exist
    when all four brought mons appeared), and the imitation weight."""
    out = {"train": [], "val": [], "test": []}
    # two streaming passes (recency weights need the global max timestamp);
    # holding every parsed battle in memory is the exact OOM data.prep avoids
    max_ts = 0
    for n, rec in enumerate(iter_battles(*files)):
        if max_battles and n >= max_battles:
            break
        max_ts = max(max_ts, rec["ts"])

    def capped():
        """Yield at most ``max_battles`` parsed records, streaming."""
        for n, rec in enumerate(iter_battles(*files)):
            if max_battles and n >= max_battles:
                return
            yield rec

    for rec in capped():
        for p in ("p1", "p2"):
            opp = "p2" if p == "p1" else "p1"
            first = next((t["states"][p] for t in rec["turns"]
                          if t["states"] is not None), None)
            last = next((t["states"][p] for t in reversed(rec["turns"])
                         if t["states"] is not None), None)
            if first is None or last is None:
                continue
            leads = sorted(m["team_idx"] for m in first["my"]["team"]
                           if m["active_slot"] is not None)
            if len(leads) != 2 or tuple(leads) not in PAIR_INDEX:
                continue
            appeared = {m["team_idx"] for m in last["my"]["team"]
                        if m["appeared"]} | set(leads)
            if len(appeared) > 4 or len(rec["teams"][p]) < 2:
                continue
            bring_t = np.zeros(N_MONS, dtype=np.float32)
            bring_m = np.zeros(N_MONS, dtype=np.float32)
            for k in appeared:
                bring_t[k], bring_m[k] = 1.0, 1.0
            if len(appeared) == 4:      # fully labeled: the rest sat out
                bring_m[:len(rec["teams"][p])] = 1.0
            my_f, opp_s = team_features(
                tok, rec["teams"][p],
                [sid(s["species"]) for s in rec["teams"][opp]])
            w = battle_weight(rec, p, max_ts, cfg) * \
                (1.0 if rec["winner"] == p else ls.nn_loser_weight)
            out[rec["split"]].append({
                "my": my_f, "opp": opp_s,
                "pair": PAIR_INDEX[tuple(leads)],
                "bring_t": bring_t, "bring_m": bring_m,
                "w": np.float32(w)})
    return out


def batches(examples, batch_size, shuffle=True, seed=0):
    """Yield stacked tensor dicts over ``examples``."""
    idx = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for c in range(0, len(idx), batch_size):
        rows = [examples[i] for i in idx[c:c + batch_size]]
        yield {
            "my": torch.from_numpy(np.stack([r["my"] for r in rows])),
            "opp": torch.from_numpy(np.stack([r["opp"] for r in rows])),
            "pair": torch.tensor([r["pair"] for r in rows]),
            "bring_t": torch.from_numpy(np.stack([r["bring_t"] for r in rows])),
            "bring_m": torch.from_numpy(np.stack([r["bring_m"] for r in rows])),
            "w": torch.tensor([float(r["w"]) for r in rows])}


def loss_terms(model, batch, ls=LSCFG):
    """Weighted (pair CE, masked bring BCE) for one batch."""
    bring, pair = model(batch["my"], batch["opp"])
    w = batch["w"] / batch["w"].mean().clamp_min(1e-8)
    ce = (F.cross_entropy(pair, batch["pair"], reduction="none") * w).mean()
    bce = F.binary_cross_entropy_with_logits(
        bring, batch["bring_t"], reduction="none")
    denom = batch["bring_m"].sum().clamp_min(1.0)
    bce = (bce * batch["bring_m"] * w.unsqueeze(1)).sum() / denom
    return ce, bce


@torch.no_grad()
def evaluate_leadnet(model, examples, ls=LSCFG, batch_size=1024):
    """Return {n, pair_top1, pair_top3, first_pair_rate} on ``examples``."""
    model.eval()
    n = top1 = top3 = first = 0
    for b in batches(examples, batch_size, shuffle=False):
        _, pair = model(b["my"], b["opp"])
        rank = pair.argsort(dim=1, descending=True)
        lbl = b["pair"].unsqueeze(1)
        top1 += (rank[:, :1] == lbl).any(dim=1).sum().item()
        top3 += (rank[:, :3] == lbl).any(dim=1).sum().item()
        first += (b["pair"] == PAIR_INDEX[(0, 1)]).sum().item()
        n += len(b["pair"])
    return {"n": n, "pair_top1": top1 / max(1, n),
            "pair_top3": top3 / max(1, n),
            "first_pair_rate": first / max(1, n)}


class NNLeadSelector:
    """Team preview via a trained ``LeadNet`` checkpoint."""

    def __init__(self, ckpt_path, tok, ls=LSCFG, device="cpu"):
        """Load the LeadNet and keep the shared tokenizer for vocab ids."""
        self.ls = ls
        self.tok = tok
        self.net = LeadNet.load(Path(ckpt_path), ls, device)
        self.net.eval()

    @torch.no_grad()
    def choose(self, tracker, belief, my_id, n_bring=None):
        """Return ``(order, info)`` like the other lead selectors."""
        ls = self.ls
        n_bring = n_bring or ls.n_bring
        my_sets = [m.set for m in tracker.sides[my_id].mons]
        my_f, opp_s = team_features(self.tok, my_sets, belief.species)
        bring, pair = self.net(torch.from_numpy(my_f).unsqueeze(0),
                               torch.from_numpy(opp_s).unsqueeze(0))
        bring, pair = bring[0], F.log_softmax(pair[0], dim=-1)
        n = len(my_sets)
        best, best_s = None, float("-inf")
        from agents.lead_switch.expert import enumerate_previews
        for lead, back in enumerate_previews(n, n_bring):
            if tuple(lead) not in PAIR_INDEX:
                continue
            chosen = list(lead) + list(back)
            s = (pair[PAIR_INDEX[tuple(lead)]].item()
                 + ls.nn_bring_lambda
                 * float(bring[chosen].mean()))
            if s > best_s:
                best, best_s = (lead, back), s
        lead, back = best
        back = tuple(sorted(back, key=lambda k: -float(bring[k])))
        info = {"kind": "nn", "score": best_s,
                "bring_logits": [round(float(v), 3) for v in bring[:n]]}
        return preview_order(lead, back), info
