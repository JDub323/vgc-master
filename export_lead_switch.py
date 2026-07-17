"""Export the lead/switch experiment variants as pile bundles.

A thin wrapper over export_agent.export that (a) presets the entrypoint for
each experiment variant and (b) copies the LeadNet checkpoint into the bundle
for the nn variant (export_agent only knows about the one main checkpoint).

Variants (all keep the frozen DUCT chooser for turn moves):
  expert   --leads expert --switch expert   (no net beyond the base model)
  value    --leads value  --switch value    (frozen value head as evaluator)
  nn       --leads nn     --switch value    (trained LeadNet for preview)

CLI: python export_lead_switch.py NAME --variant expert|value|nn
         [--ckpt PATH] [--leadnet PATH] [--pile PATH] [--notes TEXT]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("export_lead_switch.py"):
        raise SystemExit(0)

import shutil
import sys
from pathlib import Path

from config import CFG
from export_agent import export

VARIANTS = {
    "expert": (["--leads", "expert", "--switch", "expert"],
               "DUCT+expert-leads"),
    "value": (["--leads", "value", "--switch", "value"],
              "DUCT+value-leads"),
    "nn": (["--leads", "nn", "--switch", "value"],
           "DUCT+leadnet"),
}


def main(cfg=CFG):
    """CLI entry: export one experiment bundle from argv flags."""
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__)
        return

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    name = args[0]
    variant = opt("--variant", "expert")
    assert variant in VARIANTS, f"unknown --variant {variant!r}"
    flags, arch = VARIANTS[variant]
    leadnet = Path(opt("--leadnet", cfg.checkpoint_dir / "leadnet.pt"))
    if variant == "nn":
        assert leadnet.exists(), \
            f"--variant nn needs a trained LeadNet at {leadnet} " \
            "(python train_leads.py)"
        flags = flags + ["--leadnet", "artifacts/checkpoints/leadnet.pt"]
    entry = " ".join(["python", "lead_switch_server.py", "--agent", "search",
                      "--ckpt", "artifacts/checkpoints/ckpt.pt"] + flags)
    dst = export(name, kind="search", ckpt=opt("--ckpt"), pile=opt("--pile"),
                 notes=opt("--notes", f"lead/switch experiment: {variant}"),
                 architecture=arch, entrypoint=entry, cfg=cfg)
    if variant == "nn":
        shutil.copy2(leadnet, dst / "src" / "artifacts" / "checkpoints"
                     / "leadnet.pt")
        print(f"  bundled LeadNet from {leadnet}")


if __name__ == "__main__":
    main()
