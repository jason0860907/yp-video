"""Current customer artifacts. No legacy schema conversion or inferred joins."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yp_video.web.schemas import StrictModel


class Artifact(BaseModel):
    # Artifacts also contain presentation/model metadata we do not consume.
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class Event(Artifact):
    id: str
    label: str
    time: float = Field(ge=0)
    frame: int = Field(ge=0)
    xy: tuple[float, float] | None = None


class Rally(Artifact):
    index: int = Field(ge=1)
    set: int = Field(ge=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    score: int = Field(default=0, ge=0, le=100)
    winner: Literal["left", "right", "near", "far"] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Rally end must not precede start")
        return self


class Result(Artifact):
    job_id: str
    user_id: str
    match_id: str
    video_r2_key: str
    total_duration: float = Field(gt=0)
    rallies: list[Rally]
    action_events: list[Event]
    partial: bool = False

    @model_validator(mode="after")
    def unique_rallies(self):
        if len({r.index for r in self.rallies}) != len(self.rallies):
            raise ValueError("Duplicate rally indices")
        ordered = sorted(self.rallies, key=lambda r: r.start)
        if any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError(
                "Overlapping rallies are not supported by the App projection"
            )
        if any(r.end > self.total_duration + 0.1 for r in ordered):
            raise ValueError("Rally exceeds video duration")
        if any(e.time > self.total_duration + 0.1 for e in self.action_events):
            raise ValueError("Event exceeds video duration")
        return self


class Roster(Artifact):
    number: int
    name: str
    position: str
    hue: int
    player_id: str | None = None


class ClipCorrection(Artifact):
    key: str
    result: Literal["point", "loss"] | None = None
    loss_reason: str | None = None
    removed: bool
    tag_ids: list[str]
    trim_start: float | None = Field(default=None, ge=0)
    trim_end: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def trim_range(self):
        if (self.trim_start is None) != (self.trim_end is None):
            raise ValueError("A trim needs both boundaries")
        if (
            self.trim_start is not None
            and self.trim_end is not None
            and self.trim_end <= self.trim_start
        ):
            raise ValueError("Trim end must be after start")
        return self


class IdentificationCorrection(Artifact):
    result_id: str | None = None
    threshold: float | None = None
    unit_roster: dict[str, int]
    event_overrides: dict[str, int]
    removed_units: list[str]


class Corrections(Artifact):
    schema_version: Literal["6.0"]
    match_id: str
    updated_at: str
    roster: list[Roster]
    actions: list[ClipCorrection]
    scores: list[ClipCorrection]
    deleted_rally_indices: list[int]
    player_identification: IdentificationCorrection | None = None

    @model_validator(mode="after")
    def unique_keys(self):
        for values in (self.actions, self.scores):
            if len({c.key for c in values}) != len(values):
                raise ValueError("Duplicate correction keys")
        if len({r.number for r in self.roster}) != len(self.roster):
            raise ValueError("Duplicate roster numbers")
        return self


class Unit(Artifact):
    key: str
    events: list[str]


class Identification(Artifact):
    version: Literal[4]
    job_id: str
    user_id: str
    match_id: str
    units: list[Unit]


class LibraryRally(Rally):
    id: str
    match_id: str
    deleted_at: int | None = None


class Bundle(StrictModel):
    result: Result
    corrections: Corrections | None = None
    identification: Identification | None = None
    # The current per-match rows from the library /sync response. These own
    # App UUIDs and freely trimmed rally bounds, which corrections omits.
    library_rallies: list[LibraryRally] | None = None

    @property
    def rally_ids(self) -> dict[int, str]:
        return {r.index: r.id for r in self.library_rallies or []}

    @model_validator(mode="after")
    def same_match(self):
        if (
            self.corrections
            and self.corrections.match_id.lower() != self.result.match_id.lower()
        ):
            raise ValueError("Corrections belong to a different match")
        if self.identification and (
            self.identification.match_id.lower() != self.result.match_id.lower()
            or self.identification.user_id != self.result.user_id
        ):
            raise ValueError("Identification belongs to a different match/user")
        from uuid import UUID

        library = self.library_rallies or []
        if len({r.index for r in library}) != len(library):
            raise ValueError("Duplicate library rally indices")
        if any(r.match_id.lower() != self.result.match_id.lower() for r in library):
            raise ValueError("Library rallies belong to a different match")
        if {r.index for r in library} - {r.index for r in self.result.rallies}:
            raise ValueError(
                "Library contains rallies absent from this analysis version"
            )
        for r in library:
            r.id = str(UUID(r.id)).upper()
            if r.end > self.result.total_duration + 0.1:
                raise ValueError("Library rally exceeds video duration")
        if len({r.id for r in library}) != len(library):
            raise ValueError("Duplicate rally UUIDs")
        return self


Window = Literal["full_play", "to_next", "whole_rally"]
