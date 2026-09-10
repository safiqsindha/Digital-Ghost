"""The frozen 30-prompt evaluation set.

`init_eval_prompts` writes `data/eval_prompts.json` from
`configs/eval_prompts_source.yaml` exactly once. Every generation and
analysis step reads the frozen JSON file, never the source YAML — this is
what "written to a file before any run and never regenerated" means in
practice: regenerating after runs exist would silently break comparability
between runs generated against different prompt wordings.
"""

from __future__ import annotations

import json
from pathlib import Path

from digital_ghost.config import StudyConfig, load_eval_prompts_source


class EvalPromptsExistError(Exception):
    pass


def init_eval_prompts(study: StudyConfig, force: bool = False) -> Path:
    out_path = study.eval_prompts_file()
    if out_path.exists() and not force:
        raise EvalPromptsExistError(
            f"{out_path} already exists. Refusing to overwrite without --force: "
            "regenerating the eval prompt set after any run has used it breaks "
            "comparability across runs. If you're certain, pass --force and be "
            "aware existing generations/ratings tied to the old prompts should "
            "be treated as a separate study."
        )

    source = load_eval_prompts_source(study)
    payload = {
        "seed_root": study.seed_root,
        "seeds_per_prompt": study.eval.seeds_per_prompt,
        "prompts": [p.model_dump() for p in source.prompts],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def load_eval_prompts(study: StudyConfig) -> dict:
    path = study.eval_prompts_file()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `digital-ghost init-eval-prompts` first"
        )
    return json.loads(path.read_text())


def eval_prompt_seeds(study: StudyConfig, prompt_id: str) -> list[int]:
    """Deterministic generation seeds for one prompt, distinct from data-sampling seeds."""
    from digital_ghost.config import stable_int_hash

    return [
        stable_int_hash(f"{study.seed_root}:genseed:{prompt_id}:{i}") % (2**31 - 1)
        for i in range(1, study.eval.seeds_per_prompt + 1)
    ]
