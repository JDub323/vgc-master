"""Search debug instrumentation: phase profiler, root-belief monitor, and a
cProfile hook. Everything here is off (near-zero cost) unless a --debug /
--cprofile flag turns it on.

Why a phase profiler instead of only flamegraphs: the search is RPC-bound
(node sidecar over pipes), so a Python flamegraph mostly shows readline
waits. Bucketing wall time into the search's real phases (fork, sim step,
tokenize, net, damage calc) says directly which lever to pull —
sims_per_move vs batching net calls vs caching damage features. For
Python-level detail on top of that, pass --cprofile out.prof and open it
with snakeviz/speedscope, or run py-spy externally
(`py-spy record -- python scenarios.py`).
"""

import cProfile
import math
import time
from collections import Counter
from contextlib import contextmanager, nullcontext


class SearchDebug:
    """Callable phase timer: `with dbg("step"): ...`. Disabled instances
    return a nullcontext, so instrumentation stays in the hot path."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.t = Counter()
        self.n = Counter()

    def __call__(self, name):
        return self._timer(name) if self.enabled else nullcontext()

    @contextmanager
    def _timer(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1

    def report(self, wall) -> str:
        lines = [f"{'phase':12s} {'time':>8s} {'%':>5s} {'calls':>7s} {'ms/call':>8s}"]
        for name, t in self.t.most_common():
            lines.append(f"{name:12s} {t:7.2f}s {t / wall:5.0%} {self.n[name]:7d} "
                         f"{1000 * t / max(1, self.n[name]):8.2f}")
        other = wall - sum(self.t.values())
        lines.append(f"{'(python)':12s} {other:7.2f}s {other / wall:5.0%}")
        return "\n".join(lines)


@contextmanager
def maybe_cprofile(path):
    """`with maybe_cprofile(args_path_or_None):` — dumps pstats for
    snakeviz / `python -m pstats` / speedscope conversion."""
    if not path:
        yield
        return
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        prof.dump_stats(path)
        print(f"cProfile stats -> {path}  (snakeviz {path})")


# ---------------------------------------------------------------------------
# root particle monitor
# ---------------------------------------------------------------------------

def belief_data(belief, oracle=None) -> list:
    """Everything worth knowing about the root particle filter, per opponent
    mon: effective sample size, entropy, depletion counts, hard constraints,
    speed range, top particles — and, when the true sets are available
    (self-play / scenarios), where the oracle set sits in the posterior."""
    from beliefs import _set_key
    out = []
    summary = belief.summary()
    for k, sp in enumerate(belief.species):
        ws, ps = belief.weights[k], belief.particles[k]
        ess = 1.0 / sum(w * w for w in ws)
        entropy = -sum(w * math.log(w) for w in ws if w > 0)
        top = sorted(zip(ws, ps), key=lambda x: -x[0])[:3]
        c = belief.constraints[k]
        d = {"species": sp, "n_particles": len(ps), "ess": ess,
             "entropy": entropy,
             "soft_depletions": belief.soft_depletions[k],
             "hard_depletions": belief.hard_depletions[k],
             "revealed": {"moves": sorted(c["moves"]), "item": c["item"],
                          "ability": c["ability"], "mega": c["mega"],
                          "consumed": c["consumed"]},
             "spe_lo": summary[k]["spe_lo"], "spe_hi": summary[k]["spe_hi"],
             "top": [{"w": w, "item": p["item"], "ability": p["ability"],
                      "nature": p["nature"], "moves": list(p["moves"])}
                     for w, p in top]}
        if oracle is not None:
            key = _set_key(oracle[k])
            keys = [_set_key(p) for p in ps]
            i = keys.index(key) if key in keys else None
            d["oracle"] = {
                "in_prior": i is not None,
                "mass": ws[i] if i is not None else 0.0,
                "rank": (1 + sum(w > ws[i] for w in ws)) if i is not None else None}
        out.append(d)
    return out


def belief_report(belief, oracle=None) -> str:
    lines = ["root beliefs:"]
    for d in belief_data(belief, oracle):
        lines.append(
            f"  {d['species']:18s} ESS {d['ess']:6.1f}/{d['n_particles']:<4d} "
            f"H {d['entropy']:4.2f}  spe {d['spe_lo']:.0f}-{d['spe_hi']:.0f}  "
            f"depleted {d['soft_depletions']}/{d['hard_depletions']}")
        r = d["revealed"]
        if r["moves"] or r["item"] or r["ability"] or r["mega"]:
            lines.append(f"    revealed: {','.join(r['moves']) or '-'} | "
                         f"item {r['item'] or '?'}"
                         + (" (consumed)" if r["consumed"] else "")
                         + f" | ability {r['ability'] or '?'}"
                         + (" | MEGA" if r["mega"] else ""))
        for p in d["top"]:
            lines.append(f"    {p['w']:5.1%} {p['item'] or 'noitem':14s} "
                         f"{p['ability']:12s} {p['nature']:8s} "
                         + "/".join(p["moves"]))
        if "oracle" in d:
            o = d["oracle"]
            lines.append("    oracle: NOT IN PRIOR (filter can never converge)"
                         if not o["in_prior"] else
                         f"    oracle: rank {o['rank']}, mass {o['mass']:.1%}")
    return "\n".join(lines)


def root_table(dets, describe, top=5) -> str:
    """Per-determinization root statistics: where the visits went, the Q each
    action earned, and whether the determinizations agree — disagreement
    across dets is belief uncertainty showing up in the search."""
    lines = []
    for di, det in enumerate(dets):
        r = det.root
        order = sorted(range(len(r.my_actions)), key=lambda i: -r.my_n[i])[:top]
        lines.append(f"det {di} (root value {r.value:+.2f}):")
        for i in order:
            q = r.my_w[i] / r.my_n[i] if r.my_n[i] else 0.0
            lines.append(f"    N {int(r.my_n[i]):4d}  Q {q:+.2f}  "
                         f"P {r.my_p[i]:5.1%}  {describe(det, r.my_actions[i])}")
    return "\n".join(lines)
