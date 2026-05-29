#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Render every D2 architecture diagram under docs/architecture/.
# D2's theme 200 leaks two hardcoded colors (#CBA6f7 lilac strokes
# and #BAC2DE label fills) that class overrides cannot reach; we
# sed-swap them out to the palette's secondary + foreground after
# render. Palette source: ~/thermall/src/thermall/themes.py
# THERMALL_POWER_STATION.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCS="$ROOT/docs/architecture"
D2_BIN="${D2_BIN:-$HOME/.local/bin/d2}"

[ -x "$D2_BIN" ] || { echo "error: D2 not at $D2_BIN" >&2; exit 1; }

render_one() {
    local src="$1"
    local out="${src%.d2}.svg"
    echo "[render] ${src##*/} -> ${out##*/}"
    "$D2_BIN" -t 200 -l elk --pad 60 "$src" "$out"
    # Scrub theme-200 leakage:
    #   #CBA6f7 (lilac strokes / group borders) -> secondary (rust)
    #   #BAC2DE (light label fill)              -> foreground (cream)
    sed -i \
        -e 's/#CBA6f7/#e77843/g' \
        -e 's/#CBA6F7/#e77843/g' \
        -e 's/#BAC2DE/#ead1b5/g' \
        -e 's/#BAC2De/#ead1b5/g' \
        "$out"
}

# Render every .d2 file under docs/architecture/ (skips the old
# combined source if it's still present).
for src in "$DOCS"/nuthatch_data_lifecycle.d2 "$DOCS"/nuthatch_module_graph.d2; do
    if [ -f "$src" ]; then
        render_one "$src"
    fi
done

echo "[ok] done"
