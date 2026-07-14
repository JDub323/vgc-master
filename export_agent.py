"""Export one playable agent as a self-contained bundle in the shared pile.

Branches carry code; they do not carry agents — everything an agent needs at
play time (checkpoint, vocab, usage stats, dex, spreads) lives under the
gitignored artifacts/ tree. A *bundle* fixes that: it is a directory holding a
snapshot of this working tree's source, the behavior assets laid out exactly
where the config expects them, and a manifest telling the round-robin
coordinator how to run the agent as a subprocess (agent_server.py). Bundles
from incompatible branches coexist in one pile and play each other without any
code merging.

Bundle layout (pile/<name>/):
  manifest.json        entrypoint, protocol version, git provenance, notes
  src/                 working-tree snapshot (tracked + unignored files)
  src/artifacts/       vocab.json, usage_stats.json, dex.json, spreads.json
  src/artifacts/checkpoints/ckpt.pt   (search/policy agents only)

The Node install is deliberately NOT copied; the coordinator passes its own
via $VGC_NODE_DIR. Bundles are immutable — re-export under a new name.
The pile resolves as: --pile flag, then $VGC_PILE, then ../vgc-pile next to
the repo (shared by sibling worktrees).

CLI: python export_agent.py NAME [--agent search|policy|max-damage|random]
                            [--ckpt PATH] [--pile PATH] [--notes TEXT]
                            [--architecture LABEL] [--entrypoint "CMD ..."]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("export_agent.py"):
        raise SystemExit(0)

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from config import CFG

ROOT = Path(__file__).resolve().parent
KNOWN_KINDS = {"search": "DeterminizedDUCTChooser",
               "policy": "PolicyOnlyChooser",
               "max-damage": "MaxDamageChooser",
               "random": "RandomChooser"}
NEEDS_CHECKPOINT = ("search", "policy")
ASSET_FILES = ("vocab.json", "usage_stats.json", "dex.json", "spreads.json")


def pile_dir(explicit=None):
    """Resolve the pile directory: flag, then $VGC_PILE, then ../vgc-pile."""
    return Path(explicit or os.environ.get("VGC_PILE")
                or ROOT.parent / "vgc-pile")


def _git(*args):
    """Return stripped git stdout for provenance fields ('' on failure)."""
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=str(ROOT), timeout=15)
        return r.stdout.strip()
    except OSError:
        return ""


def snapshot_source(dst):
    """Copy the working tree (tracked + unignored untracked) into ``dst``.

    Uses git's index so .gitignore is respected — artifacts/, node_modules,
    and checkpoints never ride along by accident. Returns the file count."""
    listing = _git("ls-files", "--cached", "--others", "--exclude-standard")
    files = [f for f in listing.split("\n") if f]
    assert files, "git ls-files returned nothing — run from the repo checkout"
    copied = 0
    for rel in files:
        src = ROOT / rel
        if not src.is_file():
            continue          # staged deletions / submodule stubs
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied += 1
    return copied


def copy_assets(src_dir, kind, ckpt, cfg):
    """Copy behavior assets (and any checkpoint) into a bundle.

    Weights live under the gitignored ``artifacts/`` tree, so the source
    snapshot never carries them; they must be copied explicitly. A checkpoint
    is taken whenever the agent kind implies one or ``--ckpt`` names one, so an
    agent with its own architecture and its own weights can still ship a
    complete bundle. It always lands at ``artifacts/checkpoints/ckpt.pt``
    inside the bundle.

    Returns the extra entrypoint args (checkpoint path, bundle-relative)."""
    art = src_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    for asset in ASSET_FILES:
        p = cfg.artifacts_dir / asset
        if p.exists():
            shutil.copy2(p, art / asset)
    if kind not in NEEDS_CHECKPOINT and not ckpt:
        return []
    ckpt = Path(ckpt or cfg.checkpoint_dir / "ckpt_best.pt")
    missing = [str(p) for p in (ckpt, cfg.artifacts_dir / "vocab.json")
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"a '{kind}' agent needs these assets: {', '.join(missing)}")
    (art / "checkpoints").mkdir(exist_ok=True)
    shutil.copy2(ckpt, art / "checkpoints" / "ckpt.pt")
    return ["--ckpt", "artifacts/checkpoints/ckpt.pt"]


def build_manifest(name, kind, entrypoint, architecture, notes, cfg):
    """Return the manifest dict recorded next to the bundle source."""
    req = ROOT / "requirements.txt"
    return {
        "schema": 1,
        "name": name,
        "agent": kind,
        "architecture": architecture,
        "entrypoint": entrypoint,
        "protocol": 1,
        "format_id": cfg.format_id,
        "created": date.today().isoformat(),
        "git": {"commit": _git("rev-parse", "--short", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "requirements_sha256": hashlib.sha256(req.read_bytes()).hexdigest()
        if req.exists() else None,
        "notes": notes,
    }


def export(name, kind="search", ckpt=None, pile=None, notes="",
           architecture=None, entrypoint=None, cfg=CFG):
    """Write pile/<name>/ (manifest + source snapshot + assets); return it."""
    dst = pile_dir(pile) / name
    assert not dst.exists(), \
        f"bundle '{name}' already exists in the pile — bundles are immutable"
    if kind not in KNOWN_KINDS and not entrypoint:
        print(f"note: unknown agent kind {kind!r} — exporting anyway; make "
              "sure this branch's agent_server.build_chooser supports it")
    src_dir = dst / "src"
    n = snapshot_source(src_dir)
    extra = copy_assets(src_dir, kind, ckpt, cfg)
    cmd = (shlex.split(entrypoint) if entrypoint
           else ["python", "agent_server.py", "--agent", kind] + extra)
    manifest = build_manifest(
        name, kind, cmd,
        architecture or KNOWN_KINDS.get(kind, kind), notes, cfg)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported '{name}' -> {dst}")
    print(f"  {n} source files, agent={kind}, "
          f"commit={manifest['git']['commit']}"
          + (" (dirty)" if manifest["git"]["dirty"] else ""))
    print(f"  entrypoint: {' '.join(cmd)}")
    return dst


def main(cfg=CFG):
    """CLI entry: export one bundle from argv flags."""
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__)
        return

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    export(args[0], kind=opt("--agent", "search"), ckpt=opt("--ckpt"),
           pile=opt("--pile"), notes=opt("--notes", ""),
           architecture=opt("--architecture"),
           entrypoint=opt("--entrypoint"), cfg=cfg)


if __name__ == "__main__":
    main()
