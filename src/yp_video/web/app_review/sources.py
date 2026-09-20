"""Read-only sources for App review: pipeline annotations or customer artifacts."""

import json

from yp_video.contracts.action import event_id
from yp_video.core.rallies import load_rallies
from yp_video.extraction import links
from yp_video.reid.identity import load_assignments
from yp_video.web.action_annotations import annotation_state, normalize_events
from yp_video.web.action_waveform import video_metadata
from yp_video.web.r2_client import R2Client, resolve_cut

from .models import Bundle


class CustomerArtifacts(R2Client):
    @property
    def bucket(self) -> str:
        self._ensure_config()
        return self._config.get("R2_BUCKET_CUSTOMER", "")

    @property
    def configured(self) -> bool:
        self._ensure_config()
        return bool(
            self.bucket
            and self._config.get("R2_ACCESS_KEY_ID")
            and self._config.get("R2_SECRET_ACCESS_KEY")
        )

    def read_json(self, key: str) -> dict:
        response = self._get_client().get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as body:
            payload = body.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError("Artifact exceeds 20 MB")
        return json.loads(payload)


customer = CustomerArtifacts()


def safe_component(value: str) -> str:
    if not value or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in value
    ):
        raise ValueError("Invalid artifact identifier")
    return value


def cloud_matches() -> list[dict]:
    if not customer.configured:
        raise ValueError("R2_BUCKET_CUSTOMER and R2 credentials must be configured")
    entries = []
    for obj in customer.list_objects_cached("corrections/"):
        parts = obj["key"].split("/")
        if len(parts) == 3 and parts[2].endswith(".json"):
            entries.append(
                {
                    "user_id": parts[1],
                    "match_id": parts[2][:-5],
                    "updated_at": obj["last_modified"],
                }
            )
    return sorted(entries, key=lambda e: e["updated_at"], reverse=True)


def cloud_results(user: str, match: str) -> list[dict]:
    prefix = f"results/{safe_component(user)}/{safe_component(match)}/"
    return [
        o for o in customer.list_objects_cached(prefix) if o["key"].endswith(".json")
    ]


def cloud_bundle(user: str, match: str, job: str) -> Bundle:
    user, match, job = map(safe_component, (user, match, job))
    result = customer.read_json(f"results/{user}/{match}/{job}.json")
    if (result.get("user_id"), result.get("match_id"), result.get("job_id")) != (
        user,
        match,
        job,
    ):
        raise ValueError("Result identity does not match its object key")
    corrections = customer.read_json(f"corrections/{user}/{match}.json")
    bundle = Bundle.model_validate({"result": result, "corrections": corrections})
    pi = bundle.corrections.player_identification if bundle.corrections else None
    if pi and pi.result_id:
        identification = customer.read_json(
            f"reid/{user}/{match}/{safe_component(pi.result_id)}.json"
        )
        bundle = Bundle.model_validate(
            {**bundle.model_dump(), "identification": identification}
        )
    return bundle


def local_bundle(name: str) -> Bundle:
    video = resolve_cut(name)
    if video is None:
        raise ValueError("Video not found")
    state = annotation_state(video.name)
    if state.active_error:
        raise ValueError(state.active_error.detail)
    data = state.active
    meta = (
        data
        if data and data.get("fps") and data.get("num_frames")
        else video_metadata(video)
    )
    fps = float(meta["fps"])
    frames = int(meta["num_frames"])
    duration = float(meta.get("duration") or frames / fps)
    rallies = load_rallies(video.stem)
    events = normalize_events(
        video.stem,
        data["events"] if data else [],
        fps=fps,
        num_frames=frames,
        rallies=rallies,
    )
    named = load_assignments(video.stem, links.track_keys(video.stem))
    names = {name: i for i, name in enumerate(sorted(set(named.values())), 1)}
    return Bundle.model_validate(
        {
            "corrections": {
                "schema_version": "6.0",
                "match_id": video.stem,
                "updated_at": "local",
                "roster": [
                    {"number": i, "name": name, "position": "", "hue": 0}
                    for name, i in names.items()
                ],
                "actions": [],
                "scores": [],
                "deleted_rally_indices": [],
                "player_identification": {
                    "unit_roster": {},
                    "removed_units": [],
                    "event_overrides": {
                        key: names[name] for key, name in named.items()
                    },
                },
            },
            "result": {
                "job_id": "local",
                "user_id": "local",
                "match_id": video.stem,
                "video_r2_key": "",
                "total_duration": duration,
                "rallies": [
                    {
                        "index": r["rally_id"],
                        "set": 1,
                        "start": r["start"],
                        "end": r["end"],
                        "winner": r["winner"],
                    }
                    for r in rallies
                    if r["label"] == "rally"
                ],
                "action_events": [{**e, "id": event_id(e)} for e in events],
            },
        }
    )
