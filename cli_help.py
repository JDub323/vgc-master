"""Shared, dependency-free help text for the repository's script entry points."""

import sys
import textwrap


HELP = {
    "benchmark.py": """
        Archive immutable agents and run or summarize head-to-head benchmarks.

        Usage:
          python benchmark.py archive NAME [--ckpt PATH] [--notes TEXT]
          python benchmark.py list
          python benchmark.py play A [B] [options]
          python benchmark.py standings

        Commands:
          archive NAME       Save a checkpoint and its behavior assets as NAME.
          list               List available archived agents.
          play A [B]         Play a series; B defaults to baseline. Use current
                             for the live checkpoint and assets.
          standings          Print ratings from recorded benchmark results.

        Archive options:
          --ckpt PATH        Checkpoint to archive (default: ckpt_best.pt).
          --notes TEXT       Notes stored with the archive.

        Play options:
          --sims N           Override simulations per decision.
          --workers N        Parallel game workers (default: 4).
          --temp T           Action temperature (default: 0).
          --repeat N         Repeat the selected series N times (default: 1).
          --quick N          Run only N games instead of a full series.
          --teams A,B        Restrict contestant A to these team names.
          --spectate         Serve the live spectator dashboard.
          --port N           Spectator dashboard port (default: 8020).
          --depth N          Override rollout depth.
          --label TEXT       Attach a label to the result rows.
          --no-save          Do not save replay log/HTML files.
          --allow-source-drift
                             Run archives whose source hashes differ, marking
                             their results as drifted.
          -h, --help         Show this help message and exit.
    """,
    "beliefs.py": """
        Audit the opponent-set particle filter on held-out battles.

        Usage: python beliefs.py --audit [N]

        Options:
          --audit [N]        Audit up to N test battles (default: 500).
          -h, --help         Show this help message and exit.
    """,
    "build_spreads.py": """
        Download current Pikalytics nature/spread statistics and build
        artifacts/spreads.json for the configured format.

        Usage: python build_spreads.py

        Environment overrides:
          PIKA_KEY           Use an explicit Pikalytics format key.
          PIKA_DATE          Use an explicit data month in YYYY-MM form.

        Options:
          -h, --help         Show this help message and exit.
    """,
    "data.py": """
        Run the replay dataset pipeline: download logs, parse battles, and
        prepare tokenized training shards.

        Usage: python data.py [download|parse|prep|all] [resume]

        Arguments:
          download           Download the configured Hugging Face dataset.
          parse              Convert raw logs into parsed battle files.
          prep               Build model-ready NPZ shards.
          all                Run download, parse, and prep (default).
          resume             With prep, retain completed shards and continue.

        Options:
          -h, --help         Show this help message and exit.
    """,
    "env.py": """
        Build simulator assets, test/benchmark state reconstruction, or run
        the trained agent against a local Pokemon Showdown server.

        Usage:
          python env.py --dump-dex
          python env.py --benchmark [N]
          python env.py --selftest
          python env.py --live [CKPT] [--team FILE] [--n N] [--ladder]

        Options:
          --dump-dex         Write artifacts/dex.json from simulator data.
          --benchmark [N]    Benchmark save/restore for N steps (default: 2000).
          --selftest         Verify mid-battle state reconstruction.
          --live [CKPT]      Play live games; defaults to ckpt_best.pt.
          --team FILE        Packed team file for --live (default: built-in).
          --n N              Number of live games (default: 1).
          --ladder           Search the ladder instead of accepting challenges.
          -h, --help         Show this help message and exit.
    """,
    "evaluate.py": """
        Evaluate a policy checkpoint on held-out behavior-cloning data and
        compare it with max-damage and random baselines.

        Usage: python evaluate.py [CKPT] [--switches] [--aux] [--worst [N]]

        Arguments:
          CKPT               Checkpoint (default: ckpt_best.pt).

        Options:
          --switches         Print switch-probability and pruning diagnostics.
          --aux              Evaluate item, ability, and move prediction heads.
          --worst [N]        Show N worst policy errors (default: 20).
          -h, --help         Show this help message and exit.
    """,
    "observe_game.py": """
        Run search-vs-search simulator games and print decisions, beliefs, and
        mixed strategies turn by turn.

        Usage: python observe_game.py [options]

        Options:
          --ckpt PATH        Policy checkpoint (default: ckpt_best.pt).
          --games N          Number of games (default: 1).
          --teams P1 P2      Read packed teams from two files.
          --p2 random        Make player 2 choose random legal actions.
          --temp T           Search action temperature (config default).
          --step             Wait for Enter before resolving each turn.
          --debug            Print detailed search/belief diagnostics.
          --cprofile PATH    Write a cProfile report to PATH.
          -h, --help         Show this help message and exit.
    """,
    "play.py": """
        Start a local Showdown game so a human can play against a selected bot,
        with a browser dashboard showing the bot's decisions.

        Usage: python play.py [options]

        Options:
          --team NAME        Replica team name (otherwise prompts).
          --bot KIND         search, policy, max-damage, or random (prompts).
          --games N          Number of challenges to accept (default: 1).
          --ckpt PATH        Policy checkpoint (default: ckpt_best.pt).
          --debug            Enable detailed search diagnostics.
          --no-server        Use an already-running local Showdown server.
          --no-security      Internal server-launch compatibility flag.
          -h, --help         Show this help message and exit.
    """,
    "probe_policy.py": """
        Measure how a policy checkpoint treats switches, Protect, and status
        moves on held-out positions.

        Usage: python probe_policy.py [CKPT]

        Arguments:
          CKPT               Checkpoint (default: ckpt_best.pt).

        Options:
          -h, --help         Show this help message and exit.
    """,
    "scenarios.py": """
        Run tactical search correctness scenarios, mine replay endgames, or
        reconstruct one previously mined endgame decision.

        Usage: python scenarios.py [options]

        Options:
          --mine             Extract endgame candidates to endgames.json.
          --replay N         Reconstruct and analyze candidate N.
          --uniform          Ignore the checkpoint and use uniform priors.
          --debug            Print detailed search diagnostics.
          --cprofile PATH    Write a cProfile report to PATH.
          -h, --help         Show this help message and exit.
    """,
    "selfplay.py": """
        Run resumable self-play generation, training, and checkpoint gating.

        Usage: python selfplay.py [options]

        Options:
          --hours H          Stop after approximately H hours.
          --iters N          Stop after N iterations.
          --games N          Games generated per iteration (config default).
          --procs N          Generator subprocesses (config default).
          --from PATH        Initial BC checkpoint (default: ckpt_best.pt).
          --fresh            Start a new run instead of resuming sp_last.pt.
          --no-gate          Skip the evaluation gate against the prior model.
          -h, --help         Show this help message and exit.
    """,
    "teams.py": """
        List, display, validate, or mine teams used by human and bot play.

        Usage:
          python teams.py --list
          python teams.py --show NAME
          python teams.py --validate
          python teams.py --mine [N]

        Options:
          --list             List replica team names and archetypes.
          --show NAME        Print one team in Showdown export format.
          --validate         Validate every replica team with the simulator.
          --mine [N]         Print the N most common high-rated dataset teams
                             (default: 10).
          -h, --help         Show this help message and exit.
    """,
    "train.py": """
        Train or resume the behavior-cloning policy/value model from prepared
        shards, writing checkpoints and TensorBoard metrics.

        Usage: python train.py [EPOCHS]

        Arguments:
          EPOCHS             Total epoch count (default: Config.epochs).

        Options:
          -h, --help         Show this help message and exit.
    """,
}


def show_help(script, argv=None):
    """Print help and return true when ``argv`` requests it."""
    argv = sys.argv[1:] if argv is None else argv
    if "-h" not in argv and "--help" not in argv:
        return False
    print(textwrap.dedent(HELP[script]).strip())
    return True
