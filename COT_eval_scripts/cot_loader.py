# Copyright (c) 2026. CC BY-NC 4.0.
"""
Loads chain-of-thought (CoT) annotations from the
`nvidia/libero-r-datasets` HuggingFace dataset and provides
helpers for looking up CoT text by task instruction.

Each CoT entry has the structure:

    {
      "<episode_index>": {
        "episode_start_interval": [start_step, end_step],
        "segments": [
          {
            "start_step": int,
            "end_step": int,
            "content": "Instruction: <task>. \\nPlan: ...\\n
                        What I have done: ...\\n
                        Now I need to do: ...\\n",
            "updated_content": "...",
            "updated_content_w_instruction": "..."
          },
          ...
        ]
      },
      ...
    }

The functions below normalize this so the eval code can ask
"give me the CoT prompt for task X (optionally at step t)" and
get back a single string ready to paste into the user message.
"""

import json
import os
import re
import sys
from collections import defaultdict

from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/libero-r-datasets"

# Map roboverse suite names to the HF subset folder that ships CoT.
# Only libero-10-r and libero-100-r ship cot_simple.json in this dataset
# (libero_object / libero_goal / libero_spatial don't have CoT here).
SUITE_TO_SUBSET = {
    "libero_10": "libero-10-r",
}


def fetch_cot_json(suite: str) -> dict:
    """Download cot_simple.json for a suite, return parsed dict."""
    if suite not in SUITE_TO_SUBSET:
        raise ValueError(
            f"No CoT subset known for suite {suite!r}. "
            f"Available: {list(SUITE_TO_SUBSET.keys())}"
        )
    subset = SUITE_TO_SUBSET[suite]
    path = hf_hub_download(
        REPO_ID, filename=f"{subset}/cot_simple.json", repo_type="dataset"
    )
    with open(path) as f:
        return json.load(f)


def _normalize(s: str) -> str:
    """Normalize an instruction string for fuzzy matching."""
    s = s.strip().lower().rstrip(".")
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_instruction(episode: dict) -> str | None:
    """Pull the 'Instruction: ...' string out of an episode's CoT segments."""
    for seg in episode.get("segments", []):
        for fld in ("updated_content_w_instruction", "content", "updated_content"):
            v = seg.get(fld)
            if v and "Instruction:" in v:
                m = re.search(r"Instruction:\s*([^\n.]+)", v)
                if m:
                    return m.group(1).strip()
    return None


def index_by_instruction(cot_data: dict) -> dict:
    """
    Build {normalized_instruction: [list of CoT episodes]} from
    the raw cot_simple.json dict.
    """
    out = defaultdict(list)
    for k, ep in cot_data.items():
        if not isinstance(k, str) or not k.isdigit():
            continue
        ins = _extract_instruction(ep)
        if ins:
            out[_normalize(ins)].append(ep)
    return dict(out)


def extract_plan_section(cot_text: str) -> str:
    """
    Pull just the 'Plan: ...' section out of a full CoT content string,
    stripping the 'What I have done' and 'Now I need to do' parts.
    """
    m = re.search(
        r"(Plan:.*?)(?=\n\s*What I have done:|\n\s*Now I need to do:|\Z)",
        cot_text,
        re.S,
    )
    if m:
        return m.group(1).strip()
    return cot_text.strip()


def _first_filled_segment(episode: dict) -> dict | None:
    """
    Find the first segment in this episode whose CoT content is
    actually filled in (not 'TBD'). The very first segment is often
    a placeholder with 'Plan: TBD' etc.
    """
    for seg in episode.get("segments", []):
        content = (
            seg.get("updated_content_w_instruction")
            or seg.get("updated_content")
            or seg.get("content")
            or ""
        )
        if content and "TBD" not in content[:200]:
            return seg
    # Fall back to the first segment with any content.
    for seg in episode.get("segments", []):
        if (
            seg.get("updated_content_w_instruction")
            or seg.get("updated_content")
            or seg.get("content")
        ):
            return seg
    return None


def get_cot_for_task(
    indexed: dict,
    task_instruction: str,
    mode: str,
    step_t: int = 0,
) -> str:
    """
    Look up the CoT text for a given task instruction.

    Parameters
    ----------
    indexed : dict[str, list]
        Output of `index_by_instruction(...)`.
    task_instruction : str
        The bare LIBERO task instruction (e.g.
        "put the yellow and white mug in the microwave and close it").
    mode : {'plan', 'full_first', 'segment', 'off'}
        - 'off'        : return ""
        - 'plan'       : only the 'Plan: ...' section (constant per task)
        - 'full_first' : the full first-filled segment's content
        - 'segment'    : pick the segment whose [start_step, end_step]
                         contains step_t (from the first matching episode)
    step_t : int
        Current rollout step (only used by mode='segment').
    """
    if mode == "off":
        return ""

    eps = indexed.get(_normalize(task_instruction), [])
    if not eps:
        return ""
    ep = eps[0]

    if mode == "plan":
        seg = _first_filled_segment(ep)
        if not seg:
            return ""
        content = seg.get("updated_content") or seg.get("content") or ""
        return extract_plan_section(content)

    if mode == "full_first":
        seg = _first_filled_segment(ep)
        if not seg:
            return ""
        return (
            seg.get("updated_content_w_instruction")
            or seg.get("updated_content")
            or seg.get("content")
            or ""
        ).strip()

    if mode == "segment":
        chosen = None
        for seg in ep.get("segments", []):
            if seg["start_step"] <= step_t < seg["end_step"]:
                chosen = seg
                break
        if chosen is None:
            chosen = ep["segments"][-1] if ep.get("segments") else None
        if not chosen:
            return ""
        return (
            chosen.get("updated_content_w_instruction")
            or chosen.get("updated_content")
            or chosen.get("content")
            or ""
        ).strip()

    raise ValueError(f"Unknown CoT mode: {mode!r}")


def build_augmented_instruction(
    base_instr: str, cot_text: str, mode: str
) -> str:
    """
    Combine the bare task instruction with the CoT text into the final
    user-message string fed to the model.
    """
    if mode == "off" or not cot_text:
        return base_instr
    # If cot_text already contains "Instruction:" it has the task name baked in.
    if "Instruction:" in cot_text[:60]:
        return cot_text
    return f"Instruction: {base_instr}.\n{cot_text}"


# -----------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------
if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "libero_10"
    mode = sys.argv[2] if len(sys.argv) > 2 else "plan"

    print(f"Fetching cot_simple.json for {suite} ...")
    raw = fetch_cot_json(suite)
    idx = index_by_instruction(raw)
    print(f"  {len(raw)} raw entries -> {len(idx)} unique tasks")
    print()

    sample_task = next(iter(idx.keys()))
    print(f"Sample task: {sample_task!r}")
    print(f"Available CoT episodes for it: {len(idx[sample_task])}")
    print()
    print(f"--- mode='{mode}' ---")
    text = get_cot_for_task(idx, sample_task, mode=mode, step_t=0)
    print(text)
    print()
    print(f"--- final augmented instruction ---")
    print(build_augmented_instruction(sample_task, text, mode))
