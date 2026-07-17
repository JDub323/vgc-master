"""Game-log saving + a live web spectator, shared by benchmark.py and selfplay.py.

Every game played through run_game / play_selfplay_game can hand its raw
pokemon-showdown protocol stream to a GameFeed. Two things happen:

  * On finish the game is saved (by default) under artifacts/replays/<run>/ as
    both a plain .log (spectator protocol) and a self-contained .html replay
    that renders in any browser via play.pokemonshowdown.com's replay engine —
    i.e. you can rewatch it in the real Showdown client afterwards.

  * While games are in flight, an optional zero-dependency dashboard
    (http://localhost:<port>) lists every parallel game and lets you flip
    between them, watching each one's field + event log update live.

The sim's log carries `|split|SIDE` markers: the line after is the secret
(exact HP + private UI) view for that side, the line after THAT is the public
view everyone else sees. A replay is a spectator's view, so we keep the public
line and drop the secret one — the same stream Showdown records for replays.
"""

import html as htmllib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import CFG


def resolve_public(lines):
    """Collapse `|split|` triples to the public (spectator) line."""
    out, i, n = [], 0, len(lines)
    while i < n:
        if lines[i].startswith("|split|"):
            if i + 2 < n:
                out.append(lines[i + 2])   # [i+1]=secret, [i+2]=public
            i += 3
        else:
            out.append(lines[i])
            i += 1
    return out


def _slug(s):
    """Return a lowercase filesystem-safe slug."""
    return re.sub(r"[^a-z0-9._-]+", "-", str(s).lower()).strip("-")


def _rename_players(lines, side_names):
    """Relabel `|player|p1|p1||` -> the model + team that played that side."""
    out = []
    for L in lines:
        m = re.match(r"\|player\|(p[12])\|", L)
        if m and m.group(1) in side_names:
            parts = L.split("|")            # ['', 'player', 'p1', 'p1', '', '']
            parts[3] = side_names[m.group(1)]
            out.append("|".join(parts))
        else:
            out.append(L)
    return out


REPLAY_TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8" />
<!-- version 1 -->
<title>{title}</title>
<style>
html,body{{font-family:Verdana,sans-serif;font-size:10pt;margin:0;padding:0;background:#0b0f17;}}
body{{padding:12px 0;}}
.wrapper{{color:#dfe6f3;}}
</style>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">
<input type="hidden" name="replayid" value="{rid}" />
<div class="battle"></div>
<div class="battle-log"></div>
<div class="replay-controls"></div>
<div class="replay-controls-2"></div>
<h1 style="font-weight:normal;text-align:center;color:#dfe6f3">
<strong>{header}</strong></h1>
<script type="text/plain" class="battle-log-data">{log}</script>
</div>
<script defer src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
"""


def write_replay(path_stem, pub_lines, header, rid):
    """Write `<stem>.log` (protocol) and `<stem>.html` (browser replay)."""
    log_text = "\n".join(pub_lines)
    path_stem.with_suffix(".log").write_text(log_text)
    path_stem.with_suffix(".html").write_text(REPLAY_TEMPLATE.format(
        title=htmllib.escape(header) + " replay",
        header=htmllib.escape(header),
        rid=htmllib.escape(rid),
        log=log_text))


# ---------------------------------------------------------------------------
# live spectator
# ---------------------------------------------------------------------------

class GameFeed:
    """Per-game handle. run_game feeds it raw protocol; it accumulates,
    tracks turn/result for the dashboard, and saves files on finish."""

    def __init__(self, spectator, gid, meta):
        """Bind parent spectator, integer game id, and mutable metadata mapping."""
        self.sp, self.gid, self.meta = spectator, gid, meta
        self.raw = []

    def feed(self, raw_lines):
        """Append protocol strings and update turn/status; return ``None``."""
        if not raw_lines:
            return
        with self.sp.lock:
            self.raw.extend(raw_lines)
            for L in raw_lines:
                if L.startswith("|turn|"):
                    self.meta["turn"] = int(L.split("|")[2])
                elif L.startswith("|win|"):
                    self.meta["status"] = "done"

    def finish(self, winner_side):
        """Finalize winner metadata and optionally write replay files."""
        side_of = self.meta["side_of"]                 # {'a':'p1','b':'p2'}
        who = {v: k for k, v in side_of.items()}       # {'p1':'a','p2':'b'}
        res = who.get(winner_side, "tie")
        self.meta["result"] = {"a": self.meta["a"], "b": self.meta["b"]}.get(
            res, "tie")
        self.meta["winner"] = res
        self.meta["status"] = "done"
        if not self.sp.save:
            return
        with self.sp.lock:
            pub = resolve_public(self.raw)
        names = {side_of["a"]: f"{self.meta['a']} ({self.meta['team_a']})",
                 side_of["b"]: f"{self.meta['b']} ({self.meta['team_b']})"}
        pub = _rename_players(pub, names)
        header = (f"{self.meta['a']} ({self.meta['team_a']}) vs "
                  f"{self.meta['b']} ({self.meta['team_b']}) — winner: "
                  f"{self.meta['result']}")
        stem = self.sp.dir / (f"game{self.gid:03d}_"
                              f"{_slug(self.meta['team_a'])}__vs__"
                              f"{_slug(self.meta['team_b'])}")
        write_replay(stem, pub, header, f"local-{self.gid:03d}")

    def public_lines(self):
        """Return a locked public-only copy of accumulated protocol lines."""
        with self.sp.lock:
            return resolve_public(self.raw)


class Spectator:
    """Thread-safe multi-game live dashboard and replay coordinator."""

    def __init__(self, run_name, cfg=CFG, live=False, port=8020, save=True,
                 controls=None):
        """Configure output/live server and initialize feed registries.

        ``controls`` (optional) is a runner-owned object with ``state() ->
        dict`` and ``action(cmd, arg) -> str``; when present the dashboard
        shows a control panel (skip matchup, worker +/-, pause) wired to
        ``/controls.json`` and ``/control?cmd=...`` — the spectator itself
        stays a passive viewer."""
        self.dir = cfg.artifacts_dir / "replays" / _slug(run_name)
        self.run_name = run_name
        self.save = save
        self.controls = controls
        if save:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.games = {}          # gid -> meta dict (+ its GameFeed)
        self.feeds = {}
        self.lock = threading.Lock()
        self._next = 0
        self.srv = None
        if live:
            self._start(port)

    def new_game(self, a, b, team_a, team_b, side_of, fmt):
        """Create, register, and return a new ``GameFeed``."""
        with self.lock:
            gid = self._next
            self._next += 1
            meta = {"id": gid, "a": a, "b": b, "team_a": team_a,
                    "team_b": team_b, "side_of": side_of, "fmt": fmt,
                    "turn": 0, "status": "playing", "result": "", "winner": ""}
            self.games[gid] = meta
            feed = GameFeed(self, gid, meta)
            self.feeds[gid] = feed
        return feed

    # -- dashboard ----------------------------------------------------------
    def _start(self, port):
        """Start the daemon spectator HTTP server; return ``None``."""
        sp = self

        class H(BaseHTTPRequestHandler):
            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state.json"):
                    with sp.lock:
                        games = [{k: m[k] for k in
                                  ("id", "a", "b", "team_a", "team_b",
                                   "turn", "status", "result")}
                                 for m in sp.games.values()]
                    self._send(json.dumps({"run": sp.run_name,
                                           "games": games}).encode(),
                               "application/json")
                elif self.path.startswith("/game.json"):
                    m = re.search(r"id=(\d+)", self.path)
                    gid = int(m.group(1)) if m else -1
                    feed = sp.feeds.get(gid)
                    meta = sp.games.get(gid, {})
                    md = {k: meta.get(k) for k in
                          ("a", "b", "team_a", "team_b", "turn",
                           "status", "result")}
                    md["side_a"] = meta.get("side_of", {}).get("a", "p1")
                    self._send(json.dumps({
                        "meta": md,
                        "log": feed.public_lines() if feed else []}).encode(),
                        "application/json")
                elif self.path.startswith("/controls.json"):
                    body = sp.controls.state() if sp.controls else {"off": True}
                    self._send(json.dumps(body).encode(), "application/json")
                elif self.path.startswith("/control"):
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(self.path).query)
                    msg = sp.controls.action(
                        (q.get("cmd") or [""])[0], (q.get("arg") or [""])[0]) \
                        if sp.controls else "no controls attached"
                    print(f"  [dashboard] {msg}")
                    self._send(json.dumps({"msg": msg}).encode(),
                               "application/json")
                else:
                    self._send(PAGE.encode(), "text/html")

            def log_message(self, *a):
                pass

        try:
            self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        except OSError as exc:
            raise OSError(
                f"{exc} on port {port}. If nothing you started should be "
                f"using it, this is usually a stuck/orphaned listener (WSL2 "
                f"port-forward relays are a common cause) rather than a "
                f"real conflict — try a different --dash-port, or "
                f"`wsl --shutdown` from Windows to reset networking.") \
                from exc
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        # 127.0.0.1, not localhost: under WSL2 Windows resolves localhost to
        # ::1 first and the relay only forwards IPv4 loopback -> blank hang
        print(f"  spectate live at http://127.0.0.1:{port}  "
              f"(flip between games; sprites + event log update each turn)")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>vgc — spectate</title><style>
:root{--bg:#0b0f17;--card:#151d2e;--ink:#e6ecfa;--dim:#8291ad;--acc:#5eead4;
--win:#4ade80;--lose:#f87171;--line:#243149;--bar:#2a3652}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;height:100vh;display:flex}
#side{width:290px;flex:0 0 290px;border-right:1px solid var(--line);overflow:auto;padding:14px}
#main{flex:1;overflow:auto;padding:18px 22px}
h1{font-size:15px;color:var(--acc);margin-bottom:2px}
.hint{color:var(--dim);font-size:11px;margin-bottom:14px}
.g{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin-bottom:9px;cursor:pointer}
.g:hover{border-color:var(--acc)}
.g.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
.g .t{font-size:12px}.g .m{color:var(--dim);font-size:11px;margin-top:3px}
.pill{float:right;font-size:10px;padding:1px 7px;border-radius:9px;background:var(--bar);color:var(--dim)}
.pill.playing{background:#134e4a;color:var(--acc)}
.pill.win{background:#14532d;color:var(--win)}.pill.lose{background:#4c1d1d;color:var(--lose)}
.side{display:flex;gap:10px;margin:8px 0;align-items:center}
.side b{flex:0 0 150px;color:var(--dim);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mon{display:inline-flex;flex-direction:column;align-items:center;width:66px;margin-right:6px}
.mon img{width:56px;height:56px;image-rendering:pixelated;filter:drop-shadow(0 2px 5px #0009)}
.mon.dead{opacity:.3;filter:grayscale(1)}
.hp{width:52px;height:6px;border-radius:3px;background:var(--bar);overflow:hidden;margin-top:2px}
.hp i{display:block;height:100%;background:var(--win)}.hp i.low{background:var(--lose)}
.nm{font-size:9px;color:var(--dim);margin-top:2px;max-width:64px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.st{font-size:9px;color:#fbbf24}
#log{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
.ev{padding:1px 0;color:var(--ink)}
.ev.turn{color:var(--acc);margin-top:9px;border-top:1px dashed var(--line);padding-top:6px}
.ev.faint{color:var(--lose)}.ev.dim{color:var(--dim)}.ev.win{color:var(--win);font-size:15px;margin-top:10px}
.empty{color:var(--dim);margin-top:40px;text-align:center}
#ctl{display:none;background:var(--card);border:1px solid var(--acc);border-radius:9px;padding:10px 12px;margin-bottom:12px}
#ctl .t{color:var(--acc);font-size:12px}#ctl .m{color:var(--dim);font-size:11px;margin-top:3px}
#ctl button{background:var(--bar);border:1px solid var(--line);border-radius:6px;color:var(--ink);
font:inherit;font-size:11px;padding:3px 8px;margin:6px 4px 0 0;cursor:pointer}
#ctl button:hover{border-color:var(--acc)}
#ctl button.warn{color:#fbbf24}
#ctl table{width:100%;border-collapse:collapse;margin-top:8px;font-size:11px}
#ctl td{padding:1px 4px;color:var(--dim)}#ctl td:first-child{color:var(--ink)}
#ctl .paused{color:#fbbf24}
</style></head><body>
<div id="side"><h1>spectate</h1><div class="hint" id="run">…</div>
<div id="ctl">
 <div class="t" id="ctl-pair"></div>
 <div class="m" id="ctl-stats"></div>
 <div class="m" id="ctl-workers"></div>
 <div>
  <button class="warn" onclick="ctl('skip')" title="drop this matchup's queued games; in-flight games finish">skip matchup</button>
  <button onclick="ctl('workers','-1')">&minus; worker</button>
  <button onclick="ctl('workers','1')">+ worker</button>
  <button id="ctl-pause" onclick="ctl(ctlPaused?'resume':'pause')"></button>
 </div>
 <table id="ctl-standings"></table>
</div>
<div id="list"></div></div>
<div id="main"><div class="empty" id="main-empty">← pick a game to watch</div>
<div id="viewer" style="display:none">
<h1 id="vtitle"></h1><div class="hint" id="vmeta"></div>
<div id="field"></div><div id="log"></div></div></div>
<script>
const $=id=>document.getElementById(id);
let sel=null;
const sprite=s=>`https://play.pokemonshowdown.com/sprites/gen5/${s}.png`;
const spid=s=>s.toLowerCase().replace(/[^a-z0-9-]/g,"");
const FB=`if(this.src.includes('-')){this.src=this.src.replace(/-[^./]*\\.png$/,'.png')}else{this.style.visibility='hidden'}`;
function pillClass(g){if(g.status==='playing')return'playing';
  if(g.result&&g.result===g.a)return'win';if(g.result&&g.result===g.b)return'lose';return'';}
async function lobby(){
 try{const d=await(await fetch('state.json')).json();
  $('run').textContent=d.run+' · '+d.games.length+' games';
  $('list').innerHTML=d.games.map(g=>`<div class="g ${g.id===sel?'sel':''}" onclick="pick(${g.id})">
   <span class="pill ${pillClass(g)}">${g.status==='playing'?'T'+g.turn:(g.result||'done')}</span>
   <div class="t">${g.team_a} <span style="color:var(--dim)">vs</span> ${g.team_b}</div>
   <div class="m">${g.a} vs ${g.b}</div></div>`).join('');
 }catch(e){$('run').textContent='waiting for games…';}
 setTimeout(lobby,1500);
}
function pick(id){sel=id;$('main-empty').style.display='none';$('viewer').style.display='block';view();}
let ctlPaused=false;
async function ctl(cmd,arg){try{await fetch(`control?cmd=${cmd}&arg=${arg||''}`);}catch(e){}ctlTick(true);}
async function ctlTick(once){
 try{const c=await(await fetch('controls.json')).json();
  if(!c.off){
   $('ctl').style.display='block';
   ctlPaused=!!c.paused;
   $('ctl-pair').textContent=(c.pairing||'idle')+(c.plan_total>1?`  (${c.plan_done+1}/${c.plan_total})`:'');
   $('ctl-stats').textContent=c.pairing?
    `game ${c.done}/${c.total} · A ${c.wins_a}-${c.wins_b}${c.ties?'-'+c.ties:''} B · `+
    `${c.s_per_game?c.s_per_game.toFixed(0)+'s/game · ':''}`+
    `${c.eta_min!=null?'~'+c.eta_min+'min left':''}${c.paused?'  — PAUSED':''}`:'';
  $('ctl-workers').textContent=`workers ${c.workers_alive}/${c.workers_target} (live/target)`+
    (c.budget_s?` · move budget ${c.budget_s}s`:'');
   $('ctl-pause').textContent=c.paused?'resume':'pause';
   $('ctl-pause').className=c.paused?'warn':'';
   $('ctl-standings').innerHTML=(c.standings||[]).map(r=>
    `<tr><td>${r.name}</td><td>${r.rating.toFixed(0)}</td><td>${r.games}g</td><td>${r.spm.toFixed(1)}s/mv</td></tr>`).join('');
  }
 }catch(e){}
 if(!once)setTimeout(ctlTick,2000);
}
ctlTick();
// parse the public protocol into current field + a readable event feed
function render(log,meta){
 const pos={p1a:null,p1b:null,p2a:null,p2b:null};
 const ev=[];
 for(const L of log){const f=L.split('|');const tag=f[1];
  if(tag==='switch'||tag==='drag'){const id=f[2].split(':')[0];
    const sp=f[3].split(',')[0];const hp=f[4]||'100/100';
    pos[id]={sp,hp:pctOf(hp),status:stat(hp),dead:hp.includes('fnt')};
    ev.push({c:'dim',t:`↩ ${f[2]} → ${sp}`});}
  else if(tag==='move'){ev.push({c:'',t:`▸ ${f[2]} used ${f[3]}`});}
  else if(tag==='-damage'||tag==='-heal'){const id=f[2].split(':')[0];
    if(pos[id]){pos[id].hp=pctOf(f[3]);pos[id].status=stat(f[3]);pos[id].dead=f[3].includes('fnt');}}
  else if(tag==='-status'){const id=f[2].split(':')[0];if(pos[id])pos[id].status=f[3];}
  else if(tag==='faint'){const id=f[2].split(':')[0];if(pos[id]){pos[id].dead=true;pos[id].hp=0;}
    ev.push({c:'faint',t:`✖ ${f[2]} fainted`});}
  else if(tag==='turn'){ev.push({c:'turn',t:`— turn ${f[2]} —`});}
  else if(tag==='-weather'&&f[2]&&f[2]!=='none'){ev.push({c:'dim',t:`☁ ${f[2]}`});}
  else if(tag==='win'){ev.push({c:'win',t:`🏆 ${f[2]} wins`});}
 }
 const sideRow=(pid,label)=>`<div class="side"><b>${label}</b>`+
  ['a','b'].map(s=>{const m=pos[pid+s];if(!m)return'';
   return `<div class="mon ${m.dead?'dead':''}"><img src="${sprite(spid(m.sp))}" onerror="${FB}">
    <div class="hp"><i class="${m.hp<30?'low':''}" style="width:${m.hp}%"></i></div>
    <div class="nm">${m.sp}</div>${m.status&&m.status!=='fnt'?`<div class="st">${m.status}</div>`:''}</div>`;}).join('')+`</div>`;
 $('field').innerHTML=sideRow('p1',`${meta.side_a==='p1'?meta.a:meta.b} · ${meta.side_a==='p1'?meta.team_a:meta.team_b}`)
   +sideRow('p2',`${meta.side_a==='p2'?meta.a:meta.b} · ${meta.side_a==='p2'?meta.team_a:meta.team_b}`);
 $('log').innerHTML=ev.slice(-120).map(e=>`<div class="ev ${e.c}">${e.t}</div>`).join('');
 $('log').scrollTop=1e9;
}
function pctOf(s){if(s.includes('fnt'))return 0;const m=s.match(/(\d+)\/(\d+)/);
  return m?Math.round(100*(+m[1])/(+m[2])):100;}
function stat(s){const m=s.match(/\b(brn|psn|tox|par|slp|frz|fnt)\b/);return m?m[1]:'';}
async function view(){if(sel==null)return;
 try{const d=await(await fetch('game.json?id='+sel)).json();const m=d.meta;
  $('vtitle').textContent=`${m.a} (${m.team_a})  vs  ${m.b} (${m.team_b})`;
  $('vmeta').textContent=`turn ${m.turn} · ${m.status}${m.result?' · winner '+m.result:''}`;
  render(d.log,m);
 }catch(e){}
 if(sel!=null)setTimeout(view,1500);
}
lobby();
</script></body></html>"""
