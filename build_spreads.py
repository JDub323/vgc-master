"""Build artifacts/spreads.json: an objective per-species prior over (nature,
stat-point spread) for the Pokemon Champions metagame, scraped from Pikalytics'
JSON API in ONE request for the whole format.

Why this exists: the Champions team sheets in our replay dataset REDACT nature
and stat points (every parsed set is nature='serious', evs=[0]*6). So the belief
filter had no prior for the (nature, spread) latent and assumed neutral nature,
which matches ~0% of real sets (e.g. Kingambit is 88% Adamant, Aerodactyl 95%
Jolly). This gives that latent a real marginal prior per mon. It carries NO
conditioning on the specific moveset/item -- that correlation is deliberately
dropped, since the redacted data cannot estimate it anyway.

Source: https://www.pikalytics.com/api/p/<dataDate>/<format-key>/  (the trailing
slash returns the whole format as a list of per-mon objects with `natures` and
`spreads`). Spreads are already in the Champions 66-point SP system (per-stat cap
32, sum 66), so they map straight onto MAX_SP/TOTAL_SP with no conversion.

The format key (e.g. battledataregmbs3-1760) and dataDate (e.g. 2026-05) both
drift, so we DISCOVER them: the key from the format page's dropdown, the date by
probing recent months. Override with env PIKA_FORMAT / PIKA_KEY / PIKA_DATE, or
edit FORMAT below. No hardcoded per-mon data -- re-run any time to refresh.

Percentages are stored as given (Pikalytics shows a truncated tail that need not
sum to 100); beliefs.py normalizes on load and folds the missing tail into the
archetype fallback.
"""
import datetime as dt
import json
import os
import re
import urllib.request

from config import CFG

FORMAT = os.environ.get("PIKA_FORMAT", "battledataregmbs3")
UA = {"User-Agent": "Mozilla/5.0"}
MAX_SP, TOTAL_SP = 32, 66


def _sid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def discover_key(fmt):
    """Format key incl. the drifting build suffix, from the page's dropdown."""
    if os.environ.get("PIKA_KEY"):
        return os.environ["PIKA_KEY"]
    html = _get(f"https://www.pikalytics.com/pokedex/{fmt}")
    m = re.search(rf'value="({re.escape(fmt)}-\d+)"', html)
    if not m:
        raise SystemExit(f"could not find format key for {fmt} on the page")
    return m.group(1)


def candidate_dates():
    if os.environ.get("PIKA_DATE"):
        return [os.environ["PIKA_DATE"]]
    today = dt.date.today()
    out = []
    y, mo = today.year, today.month
    for _ in range(8):                       # this month back ~8 months
        out.append(f"{y:04d}-{mo:02d}")
        mo -= 1
        if mo == 0:
            y, mo = y - 1, 12
    return out


def fetch_format(fmt):
    """(dataDate, [per-mon dicts]). Probes recent months until the API returns a
    non-empty list (the API answers a bare 'false' for a date it has no data)."""
    key = discover_key(fmt)
    for date in candidate_dates():
        url = f"https://www.pikalytics.com/api/p/{date}/{key}/"
        try:
            data = json.loads(_get(url))
        except Exception:
            continue
        if isinstance(data, list) and data:
            return date, key, data
    raise SystemExit(f"no data for {fmt} (key {key}) in {candidate_dates()}")


def _parse_ev(s):
    v = [int(x) for x in s.split("/")]
    return v if len(v) == 6 and all(0 <= x <= MAX_SP for x in v) and sum(v) <= TOTAL_SP else None


def build(cfg=CFG):
    date, key, data = fetch_format(FORMAT)
    out = {"_meta": {"source": "pikalytics.com api /api/p/", "format": FORMAT,
                     "key": key, "data_date": date,
                     "fetched": dt.date.today().isoformat(),
                     "units": "champions 66-point SP (per-stat cap 32)",
                     "stat_order": ["hp", "atk", "def", "spa", "spd", "spe"]},
           "mons": {}}
    skipped = 0
    for e in data:
        natures = {_sid(n["nature"]): float(n["percent"])
                   for n in e.get("natures", []) if n.get("nature")}
        spreads = []
        for s in e.get("spreads", []):
            ev = _parse_ev(s["ev"]) if s.get("ev") else None
            if ev is not None:
                spreads.append([ev, float(s["percent"])])
        if not natures or not spreads:
            skipped += 1
            continue
        out["mons"][_sid(e["name"])] = {"natures": natures, "spreads": spreads,
                                        "games": int(e.get("games", 0) or 0)}
    p = cfg.artifacts_dir / "spreads.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}: {len(out['mons'])} mons, "
          f"{sum(len(m['spreads']) for m in out['mons'].values())} spread rows "
          f"(format {FORMAT}, key {key}, date {date}; {skipped} mons skipped: "
          f"no nature/spread data)")


if __name__ == "__main__":
    build()
