#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Launch ONE nuthatch pipeline stage as its own tmux session AND pop
# a graphical terminal showing the live log so the operator sees
# everything without ssh-ing in or running `tail -f` by hand.
#
# Usage:
#   scripts/ops/launch-stage.sh <stage> <corpus> [extra args...]
#
# <stage> is one of: ingest embed graph cluster render
# Examples:
#   scripts/ops/launch-stage.sh ingest demo
#   scripts/ops/launch-stage.sh embed demo --force
#
# Each stage is its own process. The operator inspects the output,
# decides whether to proceed, and runs this script again for the
# next stage. Deliberate per-OPERATIONS.md: chaining the whole
# pipeline is dangerously wasteful when ingest quality might be off.

set -euo pipefail

STAGE="${1:?usage: $0 <stage> <corpus> [extra args]}"
CORPUS="${2:?usage: $0 <stage> <corpus> [extra args]}"
shift 2 || true

NUTHATCH="${NUTHATCH_BIN:-.venv/bin/nuthatch}"

# Resolve corpus root so we know where logs go.
CORPUS_ROOT=$("$NUTHATCH" status --corpus "$CORPUS" 2>/dev/null \
    | awk '/^corpus:/{print $2; exit}')
if [ -z "$CORPUS_ROOT" ] || [ ! -d "$CORPUS_ROOT" ]; then
    echo "error: could not resolve corpus '$CORPUS'" >&2
    exit 2
fi

AUDIT_DIR="$CORPUS_ROOT/.kg/audit"
mkdir -p "$AUDIT_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$AUDIT_DIR/$STAGE-$TS.log"
SESSION="nuthatch-$STAGE"

# Kill any existing session for this stage so re-launch is clean.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch the stage under tmux with output unbuffered through tee.
tmux new -d -s "$SESSION" \
    "PYTHONUNBUFFERED=1 stdbuf -oL -eL '$NUTHATCH' $STAGE --corpus '$CORPUS' $* 2>&1 | stdbuf -oL tee '$LOG'"

echo "[OK] tmux session: $SESSION"
echo "     log:          $LOG"
echo "     attach:       tmux attach -t $SESSION  (Ctrl-b d to detach)"

# Pop a graphical terminal showing the live log so the operator
# doesn't have to ssh in or hunt down the log path. Best-effort:
# requires a graphical session (DISPLAY set) and gnome-terminal
# (or x-terminal-emulator) installed.
if [ -n "${DISPLAY:-}" ]; then
    TERM_TITLE="nuthatch $STAGE live log ($CORPUS)"
    if command -v gnome-terminal >/dev/null; then
        gnome-terminal --title="$TERM_TITLE" \
            -- bash -c "tail -f '$LOG'" >/dev/null 2>&1 &
        echo "[OK] launched gnome-terminal window with live log"
    elif command -v x-terminal-emulator >/dev/null; then
        x-terminal-emulator -T "$TERM_TITLE" \
            -e "tail -f '$LOG'" >/dev/null 2>&1 &
        echo "[OK] launched x-terminal-emulator window with live log"
    else
        echo "[INFO] no graphical terminal found; run 'tail -f $LOG' yourself"
    fi
else
    echo "[INFO] no DISPLAY; run 'tail -f $LOG' yourself or 'tmux attach -t $SESSION'"
fi
