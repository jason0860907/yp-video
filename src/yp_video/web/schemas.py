"""Shared request base and the vocabulary several responses share."""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict

#: The one label-status vocabulary every mode speaks (web/worklists.py).
LabelStatus = Literal["unlabeled", "pre-annotate", "in-progress", "done"]


class StrictModel(BaseModel):
    """A request body that rejects unknown fields.

    Router request models inherit this so a misspelled field fails the
    request with a 422 instead of being silently ignored into a default —
    which is how a mistyped ``btach_size`` trains at 8 for an afternoon.
    """

    model_config = ConfigDict(extra="forbid")


class Ok(TypedDict):
    """The body of a route that only acknowledges (``{"ok": true}``)."""

    ok: bool


class DoneRequest(StrictModel):
    """A mode's Done button: mark (or with false, un-mark) a video finished."""

    done: bool = True
