"""Emit the web API's OpenAPI document for the frontend's generated types.

Run after changing a route or a response model::

    make api-types

which writes ``frontend/openapi.json`` (an ignored intermediate) and
regenerates the committed ``frontend/src/types/openapi.ts`` from it
(openapi-typescript). The frontend reads every response shape from there;
do not edit it by hand.
"""

import json
from pathlib import Path

from yp_video.web.app import app

OUT = Path(__file__).parent / "frontend" / "openapi.json"


def main() -> None:
    OUT.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
