"""Play against the bot in the real Pokémon Showdown battle client.

No custom battle GUI: the pinned open-source pokemon-showdown server is
spawned locally and you play in the official client (which loads sprites,
move buttons, HP bars, statuses and battle text from play.pokemonshowdown.com
— the standard, reputable sprite source). Meanwhile a local dashboard shows
what the bot is thinking each turn: the probability it assigns to each of
YOUR likely actions, its belief about your items/speeds, and its win estimate.

Flow (`python play.py`, menus are interactive; flags skip them):
  1. Pick your team from the replica Reg M-B pool (teams.py). The bot
     secretly picks its own from the same pool (revealed after the game).
  2. Your team's export text is printed and saved to artifacts/my_team.txt —
     paste it into the client: Teambuilder -> Import from text.
  3. Open the printed client URL, pick any username, and challenge the bot
     (its username is printed) to the Champions Reg M-B format.
  4. Watch http://localhost:8010 while you play.

Flags: --team NAME --bot search|policy|max-damage|random --games N
       --ckpt PATH --no-server --debug
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("play.py"):
        raise SystemExit(0)

import asyncio
import json
import random
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import teams as teams_lib
from agents.max_damage.v1 import MaxDamageChooser
from agents.policy_only.v1 import PolicyOnlyChooser
from agents.random.v1 import RandomChooser
from config import CFG
from env import make_live_player
from search.debug import belief_data

PolicyChooser = PolicyOnlyChooser  # backward-compatible import name


# ---------------------------------------------------------------------------
# dashboard: what is the bot thinking (stdlib http server, zero deps)
# ---------------------------------------------------------------------------

STATE = {"status": "starting...", "turn": 0, "bot": "", "format": "",
         "value_history": [], "opp_pred": [], "strategy": [], "beliefs": [],
         "field": [], "result": ""}


def _sprite(species_name):
    """Return the lowercase Showdown sprite slug for a display species."""
    return re.sub(r"[^a-z0-9-]", "", species_name.lower().replace(" ", ""))


def on_decision(battle, g, info):
    """Replace dashboard state from a live game and chooser ``ChoiceInfo``."""
    me = battle.player_role
    you = "p2" if me == "p1" else "p1"
    t = g["tracker"]
    STATE["turn"] = t.turn_no
    STATE["status"] = f"turn {t.turn_no} — bot has chosen"
    STATE["value_history"] = (STATE["value_history"] + [round(info["value"], 3)])[-60:]
    STATE["opp_pred"] = [{"desc": d, "p": p} for d, p in info["opp_pred"][:6]]
    STATE["strategy"] = [{"desc": d, "p": p} for d, p in info["strategy"][:6]]
    beliefs = []
    for d, m in zip(belief_data(g["belief"]), t.sides[you].mons):
        if not m.appeared:
            continue
        beliefs.append({
            "sprite": _sprite(m.species_cur), "species": d["species"],
            "hp": round(m.hp, 3), "status": m.status, "fainted": m.fainted,
            "items": [{"item": p["item"] or "no item", "p": p["w"]}
                      for p in d["top"]],
            "spe": [round(d["spe_lo"]), round(d["spe_hi"])],
            "ess": round(d["ess"], 1)})
    STATE["beliefs"] = beliefs
    STATE["field"] = [
        {"side": "bot" if pid == me else "you",
         "mons": [{"sprite": _sprite(m.species_cur), "hp": round(m.hp, 3),
                   "status": m.status}
                  for m in t.sides[pid].mons
                  if m.active_slot is not None and not m.fainted]}
        for pid in (me, you)]


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>vgc-bot — what is it thinking?</title><style>
:root{--bg:#0f1420;--card:#1a2233;--ink:#e8edf7;--dim:#8a94ab;--acc:#5eead4;
--bad:#f87171;--bar:#334155}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 ui-monospace,Consolas,monospace;padding:24px;max-width:1060px;margin:auto}
h1{font-size:18px;color:var(--acc);margin-bottom:2px}
h2{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.12em;margin:0 0 10px}
.sub{color:var(--dim);margin-bottom:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:var(--card);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.row{display:flex;align-items:center;gap:10px;margin:5px 0}
.lbl{flex:0 0 46%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{flex:1;height:10px;background:var(--bar);border-radius:5px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#2dd4bf,#60a5fa);
border-radius:5px;transition:width .5s ease}
.pct{flex:0 0 48px;text-align:right;color:var(--acc)}
.mon{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-top:1px solid #232d42}
.mon img{width:64px;height:64px;image-rendering:pixelated;filter:drop-shadow(0 2px 6px #0008)}
.mon .hp{height:7px;border-radius:4px;background:var(--bar);margin:4px 0 6px;overflow:hidden}
.mon .hp i{display:block;height:100%;background:#4ade80;transition:width .5s}
.mon .hp i.low{background:var(--bad)}
.tag{display:inline-block;background:#3b2530;color:#fda4af;border-radius:4px;
padding:0 6px;font-size:11px;margin-left:6px}
.dead{opacity:.35;filter:grayscale(1)}
svg polyline{fill:none;stroke:var(--acc);stroke-width:2}
svg line{stroke:#2a3550;stroke-width:1}
.big{font-size:26px;color:var(--acc)}
.faint{color:var(--dim);font-size:12px}
#result{color:#fbbf24;font-size:16px;margin-top:6px}
details{margin-top:4px}summary{cursor:pointer;color:var(--dim)}
</style></head><body>
<h1>vgc-bot</h1><div class="sub" id="status">connecting…</div>
<div class="grid">
<div>
 <div class="card"><h2>Win confidence (bot's view)</h2>
  <div class="row"><span class="big" id="val">±0.00</span>
  <svg id="spark" width="100%" height="48" viewBox="0 0 300 48" preserveAspectRatio="none">
  <line x1="0" y1="24" x2="300" y2="24"/><polyline id="line" points=""/></svg></div>
  <div class="faint">-1 = you win &nbsp; +1 = bot wins &nbsp; turn <span id="turn">0</span></div>
  <div id="result"></div></div>
 <div class="card"><h2>It expects YOU to…</h2><div id="pred"></div>
  <details><summary>its own plan (spoilers)</summary><div id="plan"></div></details></div>
 <div class="card"><h2>On the field</h2><div id="field"></div></div>
</div>
<div>
 <div class="card"><h2>What it believes about your Pokémon</h2><div id="beliefs"></div></div>
</div>
</div>
<script>
const S=id=>document.getElementById(id);
const bars=(rows,max)=>rows.map(r=>`<div class="row"><span class="lbl">${r.desc??r.item}</span>
<span class="bar"><i style="width:${(100*r.p/(max||1)).toFixed(1)}%"></i></span>
<span class="pct">${(100*r.p).toFixed(0)}%</span></div>`).join("");
const sprite=s=>`https://play.pokemonshowdown.com/sprites/gen5/${s}.png`;
// unknown forme id -> retry the base species ('foo-forme' -> 'foo');
// if that fails too (no dash left), hide instead of showing a broken icon
const FALLBACK=`if(this.src.includes('-')){this.src=this.src.replace(/-[^./]*\\.png$/,'.png')}else{this.style.visibility='hidden'}`;
async function tick(){
 try{
  const d=await (await fetch("state.json")).json();
  S("status").textContent=`${d.status}  ·  vs ${d.bot}  ·  ${d.format}`;
  S("turn").textContent=d.turn;
  const v=d.value_history.at(-1)??0;
  S("val").textContent=(v>=0?"+":"")+v.toFixed(2);
  const pts=d.value_history.map((y,i)=>`${(300*i/Math.max(1,d.value_history.length-1)).toFixed(1)},${(24-22*y).toFixed(1)}`);
  S("line").setAttribute("points",pts.join(" "));
  S("pred").innerHTML=bars(d.opp_pred,Math.max(...d.opp_pred.map(r=>r.p),0.01));
  S("plan").innerHTML=bars(d.strategy,Math.max(...d.strategy.map(r=>r.p),0.01));
  S("field").innerHTML=d.field.map(side=>`<div class="row"><span class="lbl">${side.side}</span>`+
   side.mons.map(m=>`<img title="${m.status}" src="${sprite(m.sprite)}" onerror="${FALLBACK}" width="40" height="40" style="image-rendering:pixelated">`).join("")+`</div>`).join("");
  S("beliefs").innerHTML=d.beliefs.map(b=>`<div class="mon ${b.fainted?"dead":""}">
   <img src="${sprite(b.sprite)}" onerror="${FALLBACK}">
   <div style="flex:1"><b>${b.species}</b>${b.status?`<span class="tag">${b.status}</span>`:""}
   <div class="hp"><i class="${b.hp<0.3?"low":""}" style="width:${(100*b.hp).toFixed(0)}%"></i></div>
   ${bars(b.items,1)}
   <div class="faint">speed ${b.spe[0]}–${b.spe[1]} · ${b.ess} live particles</div>
   </div></div>`).join("");
  S("result").textContent=d.result;
 }catch(e){S("status").textContent="dashboard waiting for the bot… ("+e+")";}
 setTimeout(tick,1000);
}
tick();
</script></body></html>"""


def start_dashboard(port):
    """Start and return a daemon ``ThreadingHTTPServer`` on ``port``."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body, ctype = (json.dumps(STATE).encode(), "application/json") \
                if self.path.endswith("state.json") else (PAGE.encode(), "text/html")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------
# local showdown server (the open-source engine, spawned from the pinned pkg)
# ---------------------------------------------------------------------------

def port_open(port):
    """Return whether localhost accepts a TCP connection on ``port``."""
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_showdown(cfg):
    """Start the local Showdown server, or return ``None`` if already open."""
    if port_open(cfg.showdown_port):
        print(f"reusing the Showdown server already on :{cfg.showdown_port}")
        return None
    root = cfg.node_dir / "node_modules" / "pokemon-showdown"
    assert root.exists(), f"pokemon-showdown not installed under {root} (see README setup)"
    # the git install ships no logs/ or config/chat-plugins/ trees; the
    # server's repl cleanup and chat-plugin data writes crash on the missing
    # dirs (release tarballs include them)
    (root / "logs" / "repl").mkdir(parents=True, exist_ok=True)
    (root / "config" / "chat-plugins").mkdir(parents=True, exist_ok=True)
    log = open(cfg.artifacts_dir / "showdown-server.log", "w")
    proc = subprocess.Popen(
        [cfg.node_bin, "pokemon-showdown", "start", str(cfg.showdown_port),
         "--no-security"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):                     # first boot copies config, ~slow
        if port_open(cfg.showdown_port):
            return proc
        time.sleep(0.5)
    raise RuntimeError("showdown server did not come up; see artifacts/showdown-server.log")


# ---------------------------------------------------------------------------

def pick(prompt, options):
    """Prompt until valid and return the selected option's value."""
    for i, (name, note) in enumerate(options):
        print(f"  [{i}] {name:24s} {note}")
    while True:
        raw = input(f"{prompt} [0-{len(options) - 1}]: ").strip()
        if raw.isdigit() and int(raw) < len(options):
            return options[int(raw)][0]


BOTS = [("search", "full DUCT search (strongest, slowest)"),
        ("policy", "policy net only, no search"),
        ("max-damage", "greedy damage floor"),
        ("random", "uniform random floor")]


def build_chooser(kind, ckpt, cfg, debug):
    """Return the requested versioned/baseline ``MoveChooser``."""
    if kind == "random":
        return RandomChooser()
    if kind == "max-damage":
        return MaxDamageChooser(cfg)
    import torch

    from agents.determinized_duct.v1 import DeterminizedDUCTChooser
    from models.policy_value import PolicyValueNet
    from tokenizer import PositionTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    searcher = DeterminizedDUCTChooser(
        PolicyValueNet.load(ckpt, cfg, device), PositionTokenizer.load(cfg),
        cfg, debug=debug)
    return searcher if kind == "search" else PolicyOnlyChooser(searcher)


def main(cfg=CFG):
    """Orchestrate team selection, local server, dashboard, and live games."""
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    team_menu = teams_lib.menu()
    my_team = opt("--team") or pick("your team", team_menu)
    bot_kind = opt("--bot") or pick("opponent bot", BOTS)
    n_games = int(opt("--games", 1))
    rng = random.Random()
    bot_team = rng.choice([n for n, _ in team_menu])

    my_sets = teams_lib.get(my_team)
    export = teams_lib.TEAMS[my_team][1].strip()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "my_team.txt").write_text(export)

    server = None if "--no-server" in args else start_showdown(cfg)
    start_dashboard(cfg.dashboard_port)

    chooser = build_chooser(bot_kind, opt("--ckpt", cfg.checkpoint_dir / "ckpt_best.pt"),
                            cfg, "--debug" in args)
    up = cfg.artifacts_dir / "usage_stats.json"
    usage = json.loads(up.read_text()) if up.exists() else {}
    username = f"vgc-bot-{bot_kind}"[:18]
    STATE.update(bot=f"{bot_kind} bot", format=cfg.format_id,
                 status="waiting for your challenge...")
    player = make_live_player(
        teams_lib.get(bot_team), chooser, usage, cfg, on_decision=on_decision,
        account_configuration=AccountConfiguration(username, None),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{cfg.showdown_port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?"))

    print("\n" + "=" * 72)
    print(f"1. open   https://play.pokemonshowdown.com/~~localhost:{cfg.showdown_port}")
    print("   (official client, local server — pick any username)")
    print(f"2. import your team (also saved to {cfg.artifacts_dir / 'my_team.txt'}):")
    print("   Teambuilder -> Import from text, format: " + cfg.format_id)
    print(f"3. challenge  {username}  to that format (chat: /user {username})")
    print(f"4. watch the bot think:  http://localhost:{cfg.dashboard_port}")
    print("=" * 72 + f"\n\n----- your team ({my_team}) -----\n{export}\n" + "-" * 33 + "\n")

    asyncio.run(player.accept_challenges(None, n_games))
    STATE["result"] = (f"bot won {player.n_won_battles}/{player.n_finished_battles} — "
                       f"it was using '{bot_team}'")
    STATE["status"] = "finished"
    print(f"\n{STATE['result']}")
    input("enter to shut down (dashboard stays up until then) ")
    if server:
        server.terminate()


if __name__ == "__main__":
    main()
