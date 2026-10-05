"""Run a resumable, sequential Q-learning parameter sweep and plot every run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import QLearning as qlearning


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "overnight_runs"

# One-factor-at-a-time experiments make it possible to attribute changes in the
# plots to a particular parameter.  "alpha" is QLearning's control_cost.
EXPERIMENTS: tuple[tuple[str, dict[str, float]], ...] = (
    ("baseline", {}),
    ("alpha_low", {"control_cost": 0.0035}),
    ("alpha_high", {"control_cost": 0.014}),
    ("learning_rate_low", {"learning_rate": 0.0025}),
    ("learning_rate_high", {"learning_rate": 0.01}),
    ("discount_factor_low", {"discount_factor": 0.99}),
    ("discount_factor_high", {"discount_factor": 0.9995}),
    ("forcing_low", {"forcing": 5.0}),
    ("forcing_high", {"forcing": 15.0}),
)


def _settings_for(overrides: dict[str, float], *, quick: bool = False) -> qlearning.ExperimentDefaults:
    values = dict(overrides)
    forcing = values.pop("forcing", None)
    if forcing is not None:
        values.update(action_low=-forcing, action_high=forcing, u_ref=forcing)
    if quick:
        values.update(
            episodes=1,
            eval_episodes=1,
            evaluation_interval=1,
            max_steps=5,
            eval_max_steps=5,
            print_every=1,
            training_plot_samples=1,
            use_attractor_initial_state_ensemble=False,
        )
    return replace(qlearning.EXPERIMENT_DEFAULTS, **values)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives sweep data, plots, logs, and status",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume an existing sweep directory instead of creating a new one",
    )
    parser.add_argument(
        "--only", nargs="+", choices=[name for name, _ in EXPERIMENTS],
        help="Run only the named experiments",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use one five-step episode per run to verify the sweep machinery",
    )
    parser.add_argument("--list", action="store_true", help="Print the planned runs and exit")
    return parser


def main() -> int:
    args = _parser().parse_args()
    selected = [item for item in EXPERIMENTS if args.only is None or item[0] in args.only]
    if args.list:
        for name, overrides in selected:
            settings = _settings_for(overrides, quick=args.quick)
            print(
                f"{name}: alpha={settings.control_cost:g}, "
                f"learning_rate={settings.learning_rate:g}, "
                f"discount_factor={settings.discount_factor:g}, "
                f"forcing={settings.u_ref:g}"
            )
        return 0

    if args.resume:
        sweep_dir = args.resume.expanduser().resolve()
        if not sweep_dir.is_dir():
            raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        suffix = "-quick" if args.quick else ""
        sweep_dir = args.output_root.expanduser().resolve() / f"{stamp}{suffix}"
        sweep_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = sweep_dir / "manifest.json"
    manifest = {
        "sweep_directory": str(sweep_dir),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "quick": args.quick,
        "experiments": [],
    }
    failures = 0

    for index, (name, overrides) in enumerate(selected, start=1):
        settings = _settings_for(overrides, quick=args.quick)
        run_dir = sweep_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        data_path = run_dir / "q_learning_run.pkl.gz"
        completed_path = run_dir / "COMPLETED"
        record = {
            "name": name,
            "status": "pending",
            "data": str(data_path),
            "settings": settings.as_dict(),
        }
        manifest["experiments"].append(record)
        _write_json(manifest_path, manifest)

        if completed_path.exists():
            record["status"] = "already_completed"
            print(f"[{index}/{len(selected)}] Skipping completed run: {name}", flush=True)
            _write_json(manifest_path, manifest)
            continue

        print(f"[{index}/{len(selected)}] Training: {name}", flush=True)
        record["status"] = "training"
        record["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(manifest_path, manifest)
        try:
            run = qlearning.run_q_learning(settings)
            qlearning.save_q_learning_run(run, data_path)
            record["status"] = "plotting"
            _write_json(manifest_path, manifest)

            plot_command = [
                sys.executable,
                str(HERE / "PlotQLearning.py"),
                "--run-data", str(data_path),
                "--output-dir", str(sweep_dir / "plots"),
            ]
            if args.no_gif or args.quick:
                plot_command.append("--no-gif")
            subprocess.run(plot_command, cwd=HERE, check=True)

            completed_path.write_text(
                datetime.now().astimezone().isoformat(timespec="seconds") + "\n"
            )
            record["status"] = "completed"
            record["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"[{index}/{len(selected)}] Completed: {name}", flush=True)
        except Exception as error:
            failures += 1
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            (run_dir / "error.log").write_text(traceback.format_exc())
            print(f"[{index}/{len(selected)}] FAILED: {name}: {error}", flush=True)
            if args.stop_on_error:
                _write_json(manifest_path, manifest)
                break
        finally:
            _write_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["failure_count"] = failures
    _write_json(manifest_path, manifest)
    print(f"Sweep finished with {failures} failure(s): {sweep_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
