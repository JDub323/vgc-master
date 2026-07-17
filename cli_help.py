"""Shared, dependency-free help text for the repository's script entry points."""

import sys
import textwrap


HELP = {
    "agent_server.py": """
        Serve one chooser as a subprocess speaking the JSON-lines game
        protocol on stdin/stdout (used by round_robin.py; not interactive).

        Usage: python agent_server.py --agent KIND [options]

        Options:
          --agent KIND       search | policy | max-damage | random
                             (experiment branches may add kinds).
          --ckpt PATH        Checkpoint for net agents (default: ckpt_best.pt).
          --name LABEL       Display name in the ready handshake.
          --seed N           RNG seed for the chooser and forced switches.
          -h, --help         Show this help message and exit.

        Env:
          VGC_NODE_DIR       Override Config.node_dir (shared Node install).
    """,
    "export_agent.py": """
        Export the current working tree + behavior assets as one immutable,
        self-contained agent bundle in the shared pile.

        Usage: python export_agent.py NAME [options]

        Options:
          --agent KIND       search | policy | max-damage | random
                             (default: search).
          --ckpt PATH        Checkpoint to bundle (default: ckpt_best.pt).
          --pile PATH        Pile directory (default: $VGC_PILE or ../vgc-pile).
          --notes TEXT       Notes stored in the manifest.
          --architecture L   Label shown in tournament standings.
          --entrypoint CMD   Override the agent_server command line.
          -h, --help         Show this help message and exit.
    """,
    "round_robin.py": """
        Play cross-branch tournaments between exported agent bundles; each
        contestant runs as its own subprocess, so incompatible branches can
        play each other without merging.

        Usage:
          python round_robin.py list [--pile P]
          python round_robin.py play A B [options]
          python round_robin.py star [--anchor NAME] [options]
          python round_robin.py all [options]
          python round_robin.py standings [--pile P]

        Commands:
          list               Show bundles in the pile.
          play A B           One series over the replica-team pairing grid.
          star               Every bundle vs the anchor (default: baseline).
          all                Full round robin (finalists only; expensive).
          standings          Bradley-Terry table over <pile>/results.jsonl.

        Options:
          --pile PATH        Pile directory (default: $VGC_PILE or ../vgc-pile).
          --workers N        Parallel game workers (default: 2).
          --quick N          Random N-game subset instead of the full grid.
          --repeat N         Repeat the pairing grid N times (default: 1).
          --move-budget S    Enforce S seconds/move (late plays "default");
                             omitted = record timing only.
          --hang-timeout S   Silence budget before forfeit (default: 300).
          --temp T           Action temperature sent to agents (default: 0).
          --label TEXT       Tag for the run id and replay directory.
          --no-save          Do not save replay log/HTML files.
          --no-live          Headless: skip the live control dashboard that
                             play/star/all open by default (game trackers,
                             skip matchup, worker +/-, pause, standings).
          --dash-port P      Dashboard port (default: 8020).
          -h, --help         Show this help message and exit.
    """,
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

        Reports policy metrics twice: over the static mask (the historical
        numbers recorded in EXPERIMENTS.md) and over the position-legal action
        set the search's prior actually ranks. The two are not comparable.

        Usage: python evaluate.py [CKPT] [--value] [--switches] [--aux]
                                  [--worst [N]] [--no-legal]

        Arguments:
          CKPT               Checkpoint (default: ckpt_best.pt).

        Options:
          --value            Value-head quality vs the final outcome: MSE,
                             Brier, sign accuracy, calibration, confidence by
                             game phase, and the constant/HP-differential
                             floors.
          --switches         Print switch-probability and pruning diagnostics.
          --aux              Evaluate item, ability, and move prediction heads.
          --worst [N]        Show N worst policy errors (default: 20).
          --no-legal         Skip the position-legal table (saves a pass over
                             the test split).
          -h, --help         Show this help message and exit.
    """,
    "profile_selfplay.py": """
        Throughput profiler for game playing / self-play generation: plays
        real games through the self-play skeleton with the search phase
        profiler on, then reports moves/min, sims/s, per-move latency, a
        phase time table (sidecar RPC / net / tokenize / ...), and net
        batching stats. Runs a random-init baseline-architecture net when no
        checkpoint is available (throughput-representative).

        Usage: python profile_selfplay.py [options]

        Options:
          --games N          Games to play (default: 1).
          --max-decisions M  Stop after M search decisions across games.
          --sims S           Override cfg.sims_per_move.
          --dets K           Override cfg.n_determinizations.
          --policy-only      Skip simulations; play from raw net priors.
          --ckpt PATH        Checkpoint (default: ckpt_best.pt, then
                             selfplay/sp_last.pt, then random init).
          --seed N           RNG seed (default: 0).
          --cprofile PATH    Also dump cProfile stats for snakeviz/speedscope.
          -h, --help         Show this help message and exit.
    """,
    "replays.py": """
        Search saved game replays from the terminal and open them in the
        real Showdown replay player through a local HTTP server (the port
        triggers VS Code's forward/open popup). '/' on the server is a
        browsable index with the same substring search.

        Usage: python replays.py [options]

        REPL: type text to filter (run / team / agent / winner), a number
        to open, 'latest' for the newest match, 'r' to rescan, 'q' to quit.

        Options:
          --port N           Server port (default: 8030).
          --dir PATH         Replay root (default: artifacts/replays).
          --serve            Server only, no REPL (Ctrl-C stops).
          --latest           Open the newest replay immediately.
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
        Run tactical scenarios (endgame solve-to-terminal gates plus
        earlygame/midgame model diagnostics: switch-outs, weather wars,
        Contrary boost lines), mine replay endgames, or reconstruct one
        previously mined endgame decision. Diagnostics need a checkpoint
        and print NOTEs; only endgame gates can fail the suite.

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
        List, display, validate, or mine teams used by human and bot play,
        and build the self-play team pool.

        Usage:
          python teams.py --list
          python teams.py --show NAME
          python teams.py --validate
          python teams.py --mine [N]
          python teams.py --build-pool [N]
          python teams.py --import-pool FILE
          python teams.py --pool

        Options:
          --list             List replica team names and archetypes.
          --show NAME        Print one team in Showdown export format.
          --validate         Validate every replica team with the simulator.
          --mine [N]         Print the N most common high-rated dataset teams
                             (default: 10).
          --build-pool [N|all]  Mine N real Reg M-B teams from the dataset
                             (default: 30; 'all' = every distinct sheet,
                             ~2.8k, up to 3 variants per species combo),
                             fill redacted stat points from the Pikalytics
                             prior, validate, write
                             artifacts/selfplay_teams.json. selfplay.py +
                             profile_selfplay.py sample the pool
                             automatically once it exists.
          --fetch-pool [N]   Download real tournament teams (full EV/nature
                             sheets via pokepaste) from the VGenC top-teams
                             index into the pool; cached under
                             artifacts/pastes/, rate-limited. N caps the
                             count (default: everything for the format).
          --import-pool FILE Add teams from a Showdown export/backup dump
                             (any teams database exports this) to the pool.
          --pool             List the current self-play pool.
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
    "train_leads.py": """
        Train LeadNet (team-preview bring-4/lead-2 imitation) from parsed
        battles, using the baseline tokenizer's vocab and match-id splits.

        Usage: python train_leads.py [options]

        Options:
          --epochs N         Training epochs (default: LSConfig.nn_epochs).
          --batch N          Batch size (default: LSConfig.nn_batch).
          --max-battles N    Cap parsed battles read (smoke tests).
          --out PATH         Checkpoint path (default: checkpoints/leadnet.pt).
          --seed N           Torch/shuffle seed.
          -h, --help         Show this help message and exit.
    """,
    "lead_switch_server.py": """
        agent_server variant whose team-preview and forced-switch answers
        come from the exp/lead-switch selectors; move requests still go
        through the unmodified frozen chooser.

        Usage: python lead_switch_server.py [options]

        Options:
          --agent KIND       search | policy | max-damage | random.
          --ckpt PATH        Base-model checkpoint (default: ckpt_best.pt).
          --leads SEL        first4 | expert | value | nn (default: expert).
          --switch SEL       random | expert | value (default: expert).
          --leadnet PATH     LeadNet checkpoint (required for --leads nn).
          --name LABEL       Display name in the ready handshake.
          --seed N           RNG seed.
          -h, --help         Show this help message and exit.

        Env:
          VGC_NODE_DIR       Override Config.node_dir (shared Node install).
    """,
    "export_lead_switch.py": """
        Export one lead/switch experiment variant as a pile bundle
        (wraps export_agent.py; bundles LeadNet for the nn variant).

        Usage: python export_lead_switch.py NAME --variant expert|value|nn

        Options:
          --variant V        expert | value | nn (default: expert).
          --ckpt PATH        Base-model checkpoint (default: ckpt_best.pt).
          --leadnet PATH     LeadNet checkpoint (default: checkpoints/leadnet.pt).
          --pile PATH        Pile directory (default: $VGC_PILE or ../vgc-pile).
          --notes TEXT       Notes stored in the manifest.
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
