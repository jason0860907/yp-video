"""Request-model base: unknown fields are typos, not extensions."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """A request body that rejects unknown fields.

    Router request models inherit this so a misspelled field fails the
    request with a 422 instead of being silently ignored into a default —
    which is how a mistyped ``btach_size`` trains at 8 for an afternoon.
    """

    model_config = ConfigDict(extra="forbid")
