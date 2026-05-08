# Copyright (c) 2026. CC BY-NC 4.0.
"""
Wrapper script that runs CoT-augmented VLA-0 evaluation across an
entire LIBERO suite (default: libero_10) by spawning
`cot/eval_libero_cot.py` once per task.

This mirrors the layout of `run_libero_r_eval.py` but routes every
per-task subprocess through the CoT-aware eval script in this folder.

Usage examples
--------------
  # Full libero_10 sweep with CoT plan injection
  python cot/run_libero_r_cot_eval.py

  # Full libero_10 sweep with full first-segment CoT
  python cot/run_libero_r_cot_eval.py --cot-mode full_first

  # Single task (smoke test)
  python cot/run_libero_r_cot_eval.py \\
      --task LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate \\
      --num-steps 50

  # Baseline (no CoT) with same code path - useful as a sanity check
  python cot/run_libero_r_cot_eval.py --cot-mode off
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

# CoT is currently only available for libero_10 in nvidia/libero-r-datasets.
SUPPORTED_SUITES = ["libero_10"]
DEFAULT_MODEL_PATH = "./runs/vla0/model_last.pth"

EVAL_DEFAULTS = {
    "action_horizon": 1,
    "ensemble_prediction": 8,
    "start_seed": 7,
}


def banner(text, char="=", width=72):
    print(f"\n{char * width}\n  {text}\n{char * width}\n")


class Tee:
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "a", buffering=1)
        sys.stdout = self

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()


def get_tasks_for_suite(suite_name):
    from roboverse.evals.libero.eval import get_evaluation_tasks
    return get_evaluation_tasks(task_suite_name=suite_name).get(suite_name, [])


def get_eval_libero_root(model_path, action_horizon, ensemble_prediction,
                         num_steps, cot_mode):
    """Mirror eval_libero_cot.build_log_dir naming up to the suite folder."""
    log_dir = model_path
    if action_horizon != 0:
        log_dir = f"{log_dir}_ah_{action_horizon}"
    if ensemble_prediction > 1:
        log_dir = f"{log_dir}_ens_pred_{ensemble_prediction}"
    if num_steps > 0:
        log_dir = f"{log_dir}_num_steps_{num_steps}"
    log_dir = f"{log_dir}_cot_{cot_mode}_eval_libero"
    return log_dir


def main():
    parser = argparse.ArgumentParser(
        description="CoT-augmented VLA-0 evaluation on LIBERO-R",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--suite", type=str, nargs="+", choices=SUPPORTED_SUITES,
        default=["libero_10"],
    )
    parser.add_argument("--task", type=str, default=None,
                        help="Single task name (requires exactly one --suite)")
    parser.add_argument("--cot-mode", type=str, default="plan",
                        choices=["off", "plan", "full_first"])
    parser.add_argument("--action-horizon", type=int,
                        default=EVAL_DEFAULTS["action_horizon"])
    parser.add_argument("--ensemble-prediction", type=int,
                        default=EVAL_DEFAULTS["ensemble_prediction"])
    parser.add_argument("--start-seed", type=int,
                        default=EVAL_DEFAULTS["start_seed"])
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--task-id-count", type=int, default=1)
    parser.add_argument("--task-id-index", type=int, default=0)
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    args = parser.parse_args()

    # Tee stdout to a log file (mirrors run_libero_r_eval.py).
    if not args.dry_run and not args.no_log_file:
        log_root = os.path.dirname(os.path.abspath(args.model_path))
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(
            log_root,
            f"eval_libero_r_cot_{args.cot_mode}_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        Tee(log_path)
        print(f"Logging to: {log_path}")

    banner("VLA-0 CoT-Augmented Evaluation on LIBERO-R")
    print(f"  Model              : {args.model_path}")
    print(f"  Suites             : {args.suite}")
    print(f"  CoT mode           : {args.cot_mode}")
    print(f"  Action horizon     : {args.action_horizon}")
    print(f"  Ensemble prediction: {args.ensemble_prediction}")
    print(f"  Start seed         : {args.start_seed}")
    if args.num_steps > 0:
        print(f"  Num steps (cap)    : {args.num_steps}")

    # Build the eval plan.
    if args.task:
        assert len(args.suite) == 1, "--task requires exactly one --suite"
        plan = {args.suite[0]: [args.task]}
    else:
        plan = {s: get_tasks_for_suite(s) for s in args.suite}

    total = sum(len(v) for v in plan.values())
    print(f"\n  Total tasks to run : {total}")

    cot_eval_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_libero_cot.py"
    )

    # Run each task as its own subprocess (so a crash in one doesn't kill all).
    t0 = time.time()
    completed = 0
    failed = []
    for suite, tasks in plan.items():
        banner(f"Suite: {suite}  ({len(tasks)} tasks)", char="-")
        for i, task in enumerate(tasks, 1):
            print(f"\n  [{i}/{len(tasks)}]  {suite} / {task}")
            cmd = [
                sys.executable,
                cot_eval_script,
                "--model_path", args.model_path,
                "--task_suite_name", suite,
                "--task_name", task,
                "--action_horizon", str(args.action_horizon),
                "--ensemble_prediction", str(args.ensemble_prediction),
                "--start_seed", str(args.start_seed),
                "--task_id_count", str(args.task_id_count),
                "--task_id_index", str(args.task_id_index),
                "--cot_mode", args.cot_mode,
            ]
            if args.no_torch_compile:
                cmd.append("--no-torch-compile")
            if args.num_steps > 0:
                cmd += ["--num_steps", str(args.num_steps)]

            print(f"    elapsed: {dt.timedelta(seconds=int(time.time() - t0))}")
            if args.dry_run:
                print("    DRY-RUN:", " ".join(cmd))
                continue

            rc = subprocess.run(cmd).returncode
            if rc == 0:
                print("    Done")
                completed += 1
            else:
                print(f"    FAILED (rc={rc})")
                failed.append(f"{suite}/{task}")

    banner("Evaluation Complete")
    elapsed = dt.timedelta(seconds=int(time.time() - t0))
    print(f"  Total time : {elapsed}")
    print(f"  Completed  : {completed}/{total}")
    if failed:
        print(f"  Failed     : {len(failed)}")
        for f in failed:
            print(f"    - {f}")

    # Drop a small summary JSON next to the existing eval_summary_*.json
    if not args.dry_run:
        summary = {
            "timestamp": dt.datetime.now().isoformat(),
            "model_path": args.model_path,
            "cot_mode": args.cot_mode,
            "suites": args.suite,
            "settings": {
                "action_horizon": args.action_horizon,
                "ensemble_prediction": args.ensemble_prediction,
                "start_seed": args.start_seed,
                "num_steps": args.num_steps,
                "task_id_count": args.task_id_count,
                "task_id_index": args.task_id_index,
            },
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "total_elapsed": str(elapsed),
        }
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)),
            f"eval_summary_cot_{args.cot_mode}_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary JSON: {out_path}")
        # Reproduce the standard results-dir line for grep-ability.
        results_dir = get_eval_libero_root(
            args.model_path, args.action_horizon, args.ensemble_prediction,
            args.num_steps, args.cot_mode,
        )
        print(f"  Results dir : {results_dir}")


if __name__ == "__main__":
    main()
