"""Naming-scheme tests against the REAL sim: prove how pokemon-showdown
rewrites nicknames, and that every name-resolution seam in this repo survives
it. Run: python tests/test_naming.py  (needs node on PATH; see commands.txt).

The rule under test (the live-play KeyError bug): the sim rewrites nicknames
it wasn't explicitly given —
  - nickname == species id       -> display name  ('archaludon' -> 'Archaludon')
  - nickname == forme species    -> BASE species  ('Typhlosion-Hisui' -> 'Typhlosion')
so idents in requests never reliably match the set names this repo tracks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG
from data import Side, sid
from env import Sidecar, SidecarBattle, full_set, pack_team
from search.mcts import _pos_maps
from teams import TEAMS, get

FORME_TEAMS = ["sun-venusaur", "snow-gengar", "floette-balance"]  # Hisui/Alola/Eternal formes


def battle_requests(sc, sets_p1, sets_p2):
    b = SidecarBattle.create(sc, CFG.format_id, pack_team(sets_p1), pack_team(sets_p2))
    if b.requests["p1"].get("teamPreview"):
        b.step({"p1": "team 1234", "p2": "team 1234"})
    return b


def test_sim_rewrites_forme_nicknames(sc):
    """Documents the sim behavior itself: species list is fine, nicknames are
    rewritten. If this ever fails, the sim pin changed its naming rules."""
    sets = get("sun-venusaur")
    b = battle_requests(sc, sets, get("rain-archaludon"))
    idents = {p["ident"].partition(": ")[2]: p["details"].split(",")[0]
              for p in b.requests["p1"]["side"]["pokemon"]}
    b.destroy()
    assert "Typhlosion" in idents, idents            # nickname lost the forme
    assert idents["Typhlosion"] == "Typhlosion-Hisui"  # details kept it
    print("ok  sim rewrites forme nicknames (Typhlosion-Hisui -> 'Typhlosion')")


def check_side(request, names, tag):
    """Every party position must resolve to the right team index, and every
    team index must be reachable — on the actual request the sim produced."""
    name_to_idx = {n: k for k, n in enumerate(names)}
    idx_of_pos, pos_of_idx = _pos_maps(request, name_to_idx)
    party = request["side"]["pokemon"]
    for pos in range(1, len(party) + 1):
        k = idx_of_pos(pos)                      # must not raise
        want = sid(party[pos - 1]["details"].split(",")[0])
        got = sid(names[k])
        assert want == got or want.startswith(got) or got.startswith(want), \
            f"{tag}: pos {pos} ({want}) resolved to team idx {k} ({got})"
    assert len(pos_of_idx) == len(party), f"{tag}: unreachable team indices"


def test_replica_team_names(sc):
    """Display-named sets (teams.py / self-play path), all replica teams."""
    for name in TEAMS:
        sets = get(name)
        b = battle_requests(sc, sets, get("rain-archaludon"))
        check_side(b.requests["p1"], [s["name"] for s in sets], name)
        b.destroy()
    print(f"ok  {len(TEAMS)} replica teams resolve through _pos_maps")


def test_sid_named_teams(sc):
    """Belief-sampled sets are named by species id (live CTS reconstruction
    path — the path that crashed in play.py with KeyError 'typhlosion')."""
    for name in FORME_TEAMS:
        sets = [full_set({"species": sid(s["species"]), "moves": s["moves"],
                          "item": s["item"], "ability": s["ability"],
                          "nature": s["nature"]}) for s in get(name)]
        b = battle_requests(sc, sets, get("rain-archaludon"))
        check_side(b.requests["p1"], [s["name"] for s in sets], f"sid:{name}")
        b.destroy()
    print(f"ok  sid-named (belief-style) teams resolve through _pos_maps")


def test_tracker_matching():
    """data.Side.mon(): protocol idents with rewritten nicknames must find
    the right Mon (same rule, tracker seam)."""
    side = Side(get("sun-venusaur"))
    m = side.mon("Typhlosion", "Typhlosion-Hisui, L50, M")
    assert sid(m.set["species"]) == "typhlosionhisui"
    side2 = Side([full_set({"species": "typhlosionhisui", "moves": ["eruption"],
                            "item": "", "ability": "", "nature": "serious"})])
    m2 = side2.mon("Typhlosion", "Typhlosion-Hisui, L50, M")
    assert sid(m2.set["species"]) == "typhlosionhisui"
    print("ok  tracker Side.mon resolves rewritten nicknames")


if __name__ == "__main__":
    sc = Sidecar(CFG)
    test_sim_rewrites_forme_nicknames(sc)
    test_replica_team_names(sc)
    test_sid_named_teams(sc)
    test_tracker_matching()
    sc.close()
    print("all naming tests passed")
