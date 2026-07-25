"""The one shape a long-running worker reports progress in.

Three modules declared this alias privately and a fourth invented its own
``(message, fraction)`` order instead. Both are `Callable`, so wiring the odd
one into the batch-job scaffolding type-checked at every call site and failed
only in production, a second into the run. One definition, imported.

``(done, total, message)``: units finished, units expected, and the wording —
the worker owns the wording because only it knows what a unit is. ``done == 0``
marks a phase start (often followed by a long silent model load) and
``done == total`` its end; the job scaffolding pushes both past its throttle.
``total == 0`` means "nothing to do" and is not an error.
"""

from __future__ import annotations

from collections.abc import Callable

ProgressFn = Callable[[int, int, str], None]
