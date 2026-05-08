# Copyright (c) 2026. CC BY-NC 4.0.
"""
Per-task CoT-augmented evaluation of VLA-0 on a LIBERO suite.

This is a CoT-aware drop-in replacement for `eval/eval_libero.py`. The
model checkpoint is unchanged (no fine-tuning), and the inference loop
itself is unchanged - we just rewrite the `task.language` instruction
that gets fed into the VLM prompt so it includes the chain-of-thought
text from `nvidia/libero-r-datasets`.

How the injection works:
- The roboverse eval loop reads the task instruction from
  `init_libero_env(...)` and propagates it through `libero_to_rv_obs(...)`
  to the model as the `instr` kwarg.
- We monkey-patch `init_libero_env` so the returned `task_description`
  is the CoT-augmented string ("Instruction: ... Plan: ... Now I need
  to do: ...") instead of the bare task name.
- The model's `NumberSpaceOnlyProcessor` still constrains the OUTPUT
  to action-bin numbers only, so we are testing whether CoT *context*
  in the prompt changes the action predictions of the frozen VLA-0.

CoT modes:
- 'off'        : baseline (same prompt as eval/eval_libero.py).
- 'plan'       : only the per-task Plan list (constant per task).
- 'full_first' : first filled CoT segment (Plan + initial done/next).
- 'segment'    : *NOT IMPLEMENTED in this script* - would require
                 per-step instruction rewriting and a step counter
                 plumbed into roboverse.evals.libero.eval.eval_run.
                 Use 'plan' or 'full_first' for now.

Output dir mirrors eval/eval_libero.py but with `_cot_<mode>` inserted:
    {model_path}_ah_{ah}_ens_pred_{ep}_cot_{mode}_eval_libero/{suite}/{task}/
"""

import argparse
import gc
import os
import sys

import torch
from roboverse.evals.libero import eval as libero_eval_mod
from roboverse.evals.libero.eval import eval, get_evaluation_tasks  # noqa
from rv_train.train import get_pretrained_model

# Make sure we can import cot_loader from this same folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cot_loader import (  # noqa: E402
    build_augmented_instruction,
    fetch_cot_json,
    get_cot_for_task,
    index_by_instruction,
)


def install_cot_monkeypatch(suite: str, mode: str):
    """
    Replace `roboverse.evals.libero.eval.init_libero_env` with a wrapper
    that augments the returned `task_description` with CoT text.
    """
    if mode == "off":
        print("  CoT mode 'off' - no monkey patch installed (baseline).")
        return

    if mode == "segment":
        raise NotImplementedError(
            "CoT mode 'segment' (per-step dynamic CoT) is not supported "
            "by this script - use 'plan' or 'full_first' instead."
        )

    print(f"  Loading CoT annotations for {suite} (mode={mode}) ...")
    raw = fetch_cot_json(suite)
    indexed = index_by_instruction(raw)
    print(
        f"  Loaded {len(raw)} CoT entries -> {len(indexed)} unique task "
        f"instructions."
    )

    original = libero_eval_mod.init_libero_env

    def cot_init_libero_env(*args, **kwargs):
        env, init_states, max_steps, task_description = original(*args, **kwargs)
        cot_text = get_cot_for_task(indexed, task_description, mode=mode)
        if not cot_text:
            print(
                f"  [CoT WARNING] No CoT text found for task: "
                f"{task_description!r}. Falling back to bare instruction."
            )
            return env, init_states, max_steps, task_description
        augmented = build_augmented_instruction(task_description, cot_text, mode)
        print(
            f"  [CoT] Augmented instruction for {task_description!r}:\n"
            f"  ----------\n  {augmented}\n  ----------"
        )
        return env, init_states, max_steps, augmented

    libero_eval_mod.init_libero_env = cot_init_libero_env
    print("  CoT monkey patch installed on init_libero_env.")


def build_log_dir(args) -> str:
    log_dir = f"{args.model_path}"
    if args.action_horizon != 0:
        log_dir = f"{log_dir}_ah_{args.action_horizon}"
    if args.amp:
        log_dir = f"{log_dir}_amp"
    if args.generate_temperature > 0:
        log_dir = f"{log_dir}_gen_temp_{args.generate_temperature}"
    if args.ensemble_prediction > 1:
        log_dir = f"{log_dir}_ens_pred_{args.ensemble_prediction}"
    if args.ensemble_version > 1:
        log_dir = f"{log_dir}_ens_ver_{args.ensemble_version}"
        if args.ensemble_version == 2 and args.ensemble_2_weight != 0.5:
            log_dir = f"{log_dir}_ens_2_weight_{args.ensemble_2_weight}"
    if args.num_steps > 0:
        log_dir = f"{log_dir}_num_steps_{args.num_steps}"
    log_dir = f"{log_dir}_cot_{args.cot_mode}_eval_libero"
    log_dir = os.path.join(log_dir, args.task_suite_name)
    log_dir = os.path.join(log_dir, args.task_name)
    return log_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA-0 on LIBERO with CoT-augmented prompts"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, required=True)
    parser.add_argument("--start_seed", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=0)
    parser.add_argument("--save_all_data", action="store_true")
    parser.add_argument("--ensemble_prediction", type=int, default=1)
    parser.add_argument("--not_skip_evaluated", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--generate_temperature", type=float, default=0.0)
    parser.add_argument("--ensemble_version", type=int, default=1)
    parser.add_argument("--ensemble_2_weight", type=float, default=0.5)
    parser.add_argument("--task_id_index", type=int, default=0)
    parser.add_argument("--task_id_count", type=int, default=1)
    parser.add_argument(
        "--no-torch-compile", action="store_true", default=False
    )
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument(
        "--cot_mode",
        type=str,
        default="plan",
        choices=["off", "plan", "full_first"],
        help="CoT injection mode (default: plan).",
    )
    args = parser.parse_args()

    # Sanity check the task / suite.
    all_tasks = get_evaluation_tasks()
    assert args.task_suite_name in all_tasks, (
        f"Task suite {args.task_suite_name} not found in {list(all_tasks.keys())}"
    )
    assert args.task_name in all_tasks[args.task_suite_name], (
        f"Task {args.task_name} not found in suite {args.task_suite_name}"
    )

    # Install CoT monkey patch BEFORE building env (it overrides
    # init_libero_env which is called inside eval()).
    install_cot_monkeypatch(args.task_suite_name, args.cot_mode)

    # Load the model exactly as eval/eval_libero.py does.
    model, cfg = get_pretrained_model(
        args.model_path, 0, torch_compile=not args.no_torch_compile
    )
    model.eval()

    assert cfg.EXP.DATASET == "roboverse", (
        f"Dataset is {cfg.EXP.DATASET}, not roboverse"
    )
    assert cfg.EXP.MODEL in ["qwen"], (
        f"Model is {cfg.EXP.MODEL}, not qwen, "
        "if expanding must take care of action_type and action_horizon"
    )

    action_type = cfg.MODEL.QWEN.action_type
    if args.action_horizon == 0:
        action_horizon = cfg.MODEL.QWEN.horizon
    else:
        action_horizon = args.action_horizon

    enable_amp = args.amp
    other_args = {"generate_temperature": args.generate_temperature}

    def model_act(*aa, **kw):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=enable_amp
            ):
                return model(
                    *aa,
                    **kw,
                    **other_args,
                    get_loss=False,
                    get_action=True,
                )

    log_dir = build_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    print(f"  Eval output dir: {log_dir}")

    eval(
        model=model_act,
        action_type=action_type,
        cfg_path=cfg.DATALOADER.ROBOVERSE.cfg_path,
        cfg_opts=cfg.DATALOADER.ROBOVERSE.cfg_opts,
        task_name=args.task_name,
        task_suite_name=args.task_suite_name,
        log_dir=log_dir,
        save_video=True,
        seed=args.start_seed,
        action_horizon=action_horizon,
        skip_evaluated=not args.not_skip_evaluated,
        save_all_data=args.save_all_data,
        ensemble_prediction=args.ensemble_prediction,
        ensemble_2_weight=args.ensemble_2_weight,
        ensemble_version=args.ensemble_version,
        task_id_index=args.task_id_index,
        task_id_count=args.task_id_count,
        num_steps=args.num_steps,
    )

    del model
    del model_act
    gc.collect()


if __name__ == "__main__":
    main()
