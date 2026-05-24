# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m nuthatch`` to run the (eventual) CLI."""

from __future__ import annotations

import sys


def main() -> int:
    """Placeholder CLI entry point.

    The real CLI lands once the ingest pipeline and the corpus
    registry exist. Until then, surface a clear "not implemented"
    message so accidental invocations don't look like a silent
    success.
    """

    print(
        "nuthatch is in planning stage; no CLI surface is wired yet.\n"
        "See docs/SPEC.md for the architecture and docs/DECISIONS.md\n"
        "for the locked-in design choices.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
