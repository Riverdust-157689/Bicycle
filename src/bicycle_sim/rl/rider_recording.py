"""Extract an actually recorded first episode without rerunning physics."""

import json
from pathlib import Path

import numpy as np


def episode_is_successful(episode):
    """New speed objectives require qualified motion; old recordings keep survival semantics."""
    return episode["survived_horizon"] and episode.get("speed_qualified_success", True)


def extract_cohort_episode(data, env_id):
    if env_id < 0 or env_id >= data["qpos"].shape[1]:
        raise ValueError("Recorded environment index is out of range")
    mask = data["valid_first_episode"][:, env_id]
    if not mask.any():
        raise ValueError("Selected environment has no first-episode frames")
    track = {k: v[mask, env_id] for k, v in data.items() if k != "valid_first_episode"}
    track["episode_id"] = np.zeros(len(track["time_s"]), dtype=int)
    return track


def load_recorded_episode(recording, *, env_id=None, successful=False):
    recording = Path(recording)
    meta = json.loads((recording / "run.json").read_text())
    if env_id is None and not successful:
        with np.load(recording / "trajectory.npz") as data:
            return meta, {k: data[k] for k in data.files}
    if not (recording / "cohort_trajectory.npz").is_file():
        raise ValueError(
            "This run only saved one environment. Re-run probe_rider_gpu with --record-all-envs --first-episodes-only."
        )
    if successful:
        results = json.loads((recording / "results.json").read_text())
        winners = [r["env"] for r in results["episodes"] if episode_is_successful(r)]
        if not winners:
            raise ValueError("No completed successful episode exists in this recording")
        env_id = winners[0]
    with np.load(recording / "cohort_trajectory.npz") as data:
        track = extract_cohort_episode({k: data[k] for k in data.files}, env_id)
    meta.update(
        record_env=int(env_id),
        cohort_selection="first successful" if successful else "explicit environment",
    )
    return meta, track
