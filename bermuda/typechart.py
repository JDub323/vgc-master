"""Gen-9 type chart for features and the screening/opponent heuristic.

Only the sim decides real damage; this chart feeds the regression basis and
the cheap policy, so it must be right but needs no per-move exceptions.
"""

TYPES = ("Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
         "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
         "Dragon", "Dark", "Steel", "Fairy")

TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}

# attacker -> {defender: multiplier}; missing pairs are 1.0
_CHART = {
    "Normal":   {"Rock": .5, "Ghost": 0, "Steel": .5},
    "Fire":     {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2,
                 "Rock": .5, "Dragon": .5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2, "Rock": 2,
                 "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0,
                 "Flying": 2, "Dragon": .5},
    "Grass":    {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5,
                 "Ground": 2, "Flying": .5, "Bug": .5, "Rock": 2,
                 "Dragon": .5, "Steel": .5},
    "Ice":      {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2,
                 "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5,
                 "Psychic": .5, "Bug": .5, "Rock": 2, "Ghost": 0, "Dark": 2,
                 "Steel": 2, "Fairy": .5},
    "Poison":   {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5,
                 "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2,
                 "Flying": 0, "Bug": .5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2,
                 "Rock": .5, "Steel": .5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0,
                 "Steel": .5},
    "Bug":      {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5,
                 "Flying": .5, "Psychic": 2, "Ghost": .5, "Dark": 2,
                 "Steel": .5, "Fairy": .5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5,
                 "Flying": 2, "Bug": 2, "Steel": .5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon":   {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark":     {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5,
                 "Fairy": .5},
    "Steel":    {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2,
                 "Rock": 2, "Steel": .5, "Fairy": 2},
    "Fairy":    {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2,
                 "Dark": 2, "Steel": .5},
}


def effectiveness(attack_type, defender_types):
    """Combined multiplier of one attacking type into a type combination."""
    row = _CHART.get(attack_type, {})
    mult = 1.0
    for t in defender_types:
        mult *= row.get(t, 1.0)
    return mult


def best_offense(attack_types, defender_types, stab_types=()):
    """Best multiplier over available attacking types, STAB included."""
    best = 0.0
    for t in attack_types:
        m = effectiveness(t, defender_types)
        if t in stab_types:
            m *= 1.5
        best = max(best, m)
    return best
