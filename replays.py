"""Terminal search over saved game replays + a local server that opens them
in the real Showdown replay player.

Every game saved by spectate.py lands under ``artifacts/replays/<run>/`` as a
self-contained ``.html`` (rendered by play.pokemonshowdown.com's replay
engine) plus the raw ``.log`` protocol. Once there are a few hundred of them,
finding "that snow game where the baseline lost" by hand is hopeless — this
is the sieve:

  * a terminal REPL: type a substring (run name, team, agent, winner) to
    filter, a number to open. Newest first.
  * a zero-dependency HTTP server rooted at the replay directory. Opening a
    replay prints its ``http://localhost:<port>/...`` URL — when the port
    comes up, VS Code offers to forward/open it, so a replay is one click
    from the terminal. ``/`` serves a browsable index of every replay with
    the same substring search, so the browser side works standalone too.

CLI: python replays.py [options]
     --port N      Server port (default: 8030).
     --dir PATH    Replay root (default: artifacts/replays).
     --serve       Server only, no terminal REPL (Ctrl-C stops).
     --latest      Open the newest replay immediately, then REPL.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("replays.py"):
        raise SystemExit(0)

import html as htmllib
import re
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from config import CFG

_META_CACHE = {}   # path -> (mtime, meta dict)


def replay_meta(path):
    """Parse one replay's display metadata (header, winner, turns), cached
    by mtime so repeated scans stay cheap."""
    mtime = path.stat().st_mtime
    hit = _META_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    header = winner = ""
    turns = 0
    m = re.search(r"<strong>(.*?)</strong>", path.read_text(errors="ignore"),
                  re.S)
    if m:
        header = htmllib.unescape(m.group(1)).strip()
        wm = re.search(r"winner:\s*(.+)$", header)
        winner = wm.group(1).strip() if wm else ""
    log = path.with_suffix(".log")
    if log.exists():
        tm = re.findall(r"^\|turn\|(\d+)", log.read_text(errors="ignore"),
                        re.M)
        turns = int(tm[-1]) if tm else 0
    meta = {"header": header, "winner": winner, "turns": turns,
            "mtime": mtime}
    _META_CACHE[path] = (mtime, meta)
    return meta


def scan(root):
    """Return every replay under ``root`` as metadata dicts, newest first."""
    out = []
    for p in Path(root).rglob("*.html"):
        meta = replay_meta(p)
        out.append({"path": p, "rel": p.relative_to(root).as_posix(),
                    "run": p.parent.relative_to(root).as_posix(),
                    "game": p.stem, **meta})
    return sorted(out, key=lambda e: -e["mtime"])


def _age(secs):
    """Compact age string for a listing row."""
    for unit, div in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= div:
            return f"{int(secs / div)}{unit}"
    return f"{int(secs)}s"


INDEX_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>vgc — replays</title><style>
:root{--bg:#0b0f17;--card:#151d2e;--ink:#e6ecfa;--dim:#8291ad;--acc:#5eead4;--line:#243149}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:22px;max-width:1100px;margin:auto}
h1{font-size:16px;color:var(--acc);margin-bottom:4px}
.hint{color:var(--dim);font-size:11px;margin-bottom:12px}
input{width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;
color:var(--ink);font:inherit;padding:8px 12px;margin-bottom:14px;outline:none}
input:focus{border-color:var(--acc)}
a.row{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:8px 12px;margin-bottom:7px;color:var(--ink);text-decoration:none}
a.row:hover{border-color:var(--acc)}
.run{color:var(--acc);font-size:11px}.meta{color:var(--dim);font-size:11px;float:right}
.win{color:#4ade80}
</style></head><body>
<h1>replays</h1><div class="hint">__COUNT__ saved games — type to filter, click to watch</div>
<input id="q" placeholder="filter: run / team / agent / winner..." autofocus>
<div id="list">__ROWS__</div>
<script>
const q=document.getElementById('q');
q.addEventListener('input',()=>{const t=q.value.toLowerCase();
 for(const r of document.querySelectorAll('a.row'))
  r.style.display=r.dataset.k.includes(t)?'':'none';});
</script></body></html>"""


def index_html(entries):
    """Render the browsable index page for the current replay set."""
    rows = []
    for e in entries:
        key = htmllib.escape(f"{e['run']} {e['header']}".lower(), quote=True)
        rows.append(
            f'<a class="row" data-k="{key}" href="/{quote(e["rel"])}">'
            f'<span class="meta">{e["turns"]} turns · '
            f'{_age(time.time() - e["mtime"])} ago</span>'
            f'<span class="run">{htmllib.escape(e["run"])}</span><br>'
            f'{htmllib.escape(e["header"] or e["game"])}</a>')
    return INDEX_PAGE.replace("__COUNT__", str(len(entries))) \
                     .replace("__ROWS__", "\n".join(rows))


def serve(root, port):
    """Start the daemon replay file server; return the server object."""
    root = str(root)

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = index_html(scan(root)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def show(entries, limit=20):
    """Print the numbered listing for the current filter."""
    if not entries:
        print("  no replays match")
        return
    for i, e in enumerate(entries[:limit]):
        head = e["header"] or e["game"]
        print(f"  [{i:2d}] {_age(time.time() - e['mtime']):>4s}  "
              f"{e['run']:24.24s} {e['turns']:3d}T  {head}")
    if len(entries) > limit:
        print(f"  ... {len(entries) - limit} more (narrow the filter)")


def open_replay(entry, port):
    """Print (and try to open) one replay's URL; return None."""
    url = f"http://localhost:{port}/{quote(entry['rel'])}"
    print(f"  -> {url}")
    try:                      # native run: pops the browser; WSL/ssh: URL +
        webbrowser.open(url)  # VS Code's port-forward popup do the job
    except Exception:
        pass


def repl(root, port):
    """Interactive loop: substring filters, number opens, q quits."""
    entries = scan(root)
    filt = ""
    print(f"{len(entries)} replays under {root}")
    print(f"index: http://localhost:{port}/   "
          "(type to filter, number to open, 'r' rescan, 'q' quit)")
    show(entries)
    while True:
        try:
            raw = input(f"replays[{filt or '*'}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if raw == "q":
            return
        if raw == "r":
            entries = scan(root)
        elif raw == "latest":
            view = _filtered(entries, filt)
            if view:
                open_replay(view[0], port)
            continue
        elif raw.isdigit():
            view = _filtered(entries, filt)
            i = int(raw)
            if i < len(view):
                open_replay(view[i], port)
            else:
                print("  no such index")
            continue
        else:
            filt = raw
        show(_filtered(entries, filt))


def _filtered(entries, filt):
    """Entries whose run/header matches the case-insensitive filter."""
    if not filt:
        return entries
    t = filt.lower()
    return [e for e in entries
            if t in f"{e['run']} {e['header']} {e['game']}".lower()]


def main(cfg=CFG):
    """CLI entry: start the server and hand over to the REPL (or idle)."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    root = Path(opt("--dir", cfg.artifacts_dir / "replays"))
    port = int(opt("--port", 8030))
    if not root.exists():
        print(f"no replay directory at {root} — play a saved series first "
              "(round_robin.py / benchmark.py / selfplay.py write replays "
              "unless --no-save)")
        return
    serve(root, port)
    if "--latest" in args:
        entries = scan(root)
        if entries:
            open_replay(entries[0], port)
    if "--serve" in args:
        print(f"serving {root} at http://localhost:{port}/ (Ctrl-C stops)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    repl(root, port)


if __name__ == "__main__":
    main()
