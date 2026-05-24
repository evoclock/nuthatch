# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m nuthatch`` to run the CLI."""

from __future__ import annotations

from nuthatch.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
