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

# <stage> may be a single word (top-level subcommand, e.g. `ingest`)
# OR a quoted multi-word path for nested subcommands (e.g. `"eval cluster"`,
# `"viz d3"`). The script splits on whitespace and forwards each token
# as a separate arg to nuthatch, so:
#
#   launch-stage.sh ingest inputs                  -> nuthatch ingest --corpus inputs
#   launch-stage.sh "eval cluster" inputs --communities sbm
#                                                  -> nuthatch eval cluster --corpus inputs --communities sbm
#
# This keeps the script CLI-structure-agnostic (no hardcoded list of
# parent subcommands) while supporting the nested-subparser pattern
# nuthatch uses for `eval`, `corpus`, `viz`.
STAGE="${1:?usage: $0 <stage> <corpus> [--session-suffix S] [--python PATH] [extra args]}"
CORPUS="${2:?usage: $0 <stage> <corpus> [--session-suffix S] [--python PATH] [extra args]}"
shift 2 || true
# Split the stage string on whitespace into an array of subcommand
# tokens. Word-splitting is intentional here (the user passed the
# space-separated path inside the quoted argument); shellcheck noise
# is acceptable.
# shellcheck disable=SC2206
STAGE_ARGS=( $STAGE )

# Peel script-level flags off the front before we forward the rest to
# `nuthatch <stage>`. Two flags supported here:
#   --session-suffix S  Append `-S` to the tmux session + log filename
#                       so multiple instances of the same stage (e.g.
#                       three cluster runs with different --backend
#                       values) can co-exist without clobbering each
#                       other's session or log. Without this flag the
#                       script kills any prior `nuthatch-$STAGE` session
#                       on launch (existing behaviour).
#   --python PATH       Use a different python interpreter for this
#                       run. Needed for `nuthatch cluster --backend sbm`
#                       which requires graph-tool from a conda env
#                       (e.g. nuthatch-gt); other backends + stages
#                       stay on the default `.venv`.
SUFFIX=""
PYTHON_OVERRIDE=""
FORWARD=()
while [ $# -gt 0 ]; do
    case "$1" in
        --session-suffix)
            SUFFIX="${2:?--session-suffix needs a value}"
            shift 2
            ;;
        --python)
            PYTHON_OVERRIDE="${2:?--python needs a value}"
            shift 2
            ;;
        *)
            FORWARD+=("$1")
            shift
            ;;
    esac
done

# Resolve which nuthatch invocation to use. Default is the venv's
# console script; --python swaps to `<python> -m nuthatch` so a conda
# env (with its own python + nuthatch install) can run the stage
# without re-pointing NUTHATCH_BIN globally.
if [ -n "$PYTHON_OVERRIDE" ]; then
    NUTHATCH_INVOKE=("$PYTHON_OVERRIDE" -m nuthatch)
else
    NUTHATCH_INVOKE=("${NUTHATCH_BIN:-.venv/bin/nuthatch}")
fi

# Resolve corpus root so we know where logs go. Use the same invocation
# the stage will use so we know the corpus is resolvable from that env.
CORPUS_ROOT=$("${NUTHATCH_INVOKE[@]}" status --corpus "$CORPUS" 2>/dev/null \
    | awk '/^corpus:/{print $2; exit}')
if [ -z "$CORPUS_ROOT" ] || [ ! -d "$CORPUS_ROOT" ]; then
    echo "error: could not resolve corpus '$CORPUS'" >&2
    exit 2
fi

AUDIT_DIR="$CORPUS_ROOT/.kg/audit"
mkdir -p "$AUDIT_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
# Sanitise the stage path for filenames + tmux session names: replace
# any internal whitespace with dashes so "eval cluster" -> "eval-cluster"
# in `nuthatch-eval-cluster` / `eval-cluster-<ts>.log`. Then apply the
# optional --session-suffix.
STAGE_SLUG="${STAGE// /-}"
NAME_TAG="$STAGE_SLUG${SUFFIX:+-$SUFFIX}"
LOG="$AUDIT_DIR/$NAME_TAG-$TS.log"
SESSION="nuthatch-$NAME_TAG"

# Kill any existing session for this exact session name so re-launch is
# clean. With --session-suffix this only kills the matching suffixed
# session, leaving any parallel runs untouched.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch the stage under tmux with output unbuffered through tee. Build
# the invocation as a shell-escaped string so the tmux command stays
# quoting-safe even when forwarded args contain spaces.
escape_arg() { printf "%q" "$1"; }
INVOKE_STR=""
for tok in "${NUTHATCH_INVOKE[@]}"; do
    INVOKE_STR+=" $(escape_arg "$tok")"
done
for stage_tok in "${STAGE_ARGS[@]}"; do
    INVOKE_STR+=" $(escape_arg "$stage_tok")"
done
INVOKE_STR+=" --corpus $(escape_arg "$CORPUS")"
for tok in "${FORWARD[@]:-}"; do
    [ -n "$tok" ] && INVOKE_STR+=" $(escape_arg "$tok")"
done

tmux new -d -s "$SESSION" \
    "PYTHONUNBUFFERED=1 stdbuf -oL -eL${INVOKE_STR} 2>&1 | stdbuf -oL tee $(escape_arg "$LOG")"

echo "[OK] tmux session: $SESSION"
echo "     log:          $LOG"
echo "     attach:       tmux attach -t $SESSION  (Ctrl-b d to detach)"

# Pop a graphical terminal showing the live log so the operator
# doesn't have to ssh in or hunt down the log path. Cross-platform
# best-effort: macOS via osascript, Linux via gnome-terminal /
# x-terminal-emulator / xterm, Windows via wt.exe / cmd. Falls back
# to a print-the-command line when no graphical terminal is
# detectable so headless / SSH / WSL operators still know what to
# run by hand.
TERM_TITLE="nuthatch $STAGE live log ($CORPUS)"
TAIL_CMD="tail -f '$LOG'"

launched_terminal=0
case "$(uname -s)" in
    Darwin)
        # macOS: prefer iTerm2 if open, fall back to Terminal.app.
        # AppleScript handles quoting; we pass the tail command in.
        if command -v osascript >/dev/null; then
            if osascript -e 'tell application "System Events" to (name of processes) contains "iTerm2"' 2>/dev/null | grep -q true; then
                osascript -e "tell application \"iTerm2\" to create window with default profile command \"$TAIL_CMD\"" >/dev/null 2>&1 \
                    && { echo "[OK] launched iTerm2 window with live log"; launched_terminal=1; }
            fi
            if [ "$launched_terminal" -eq 0 ]; then
                osascript -e "tell application \"Terminal\" to do script \"$TAIL_CMD\"" >/dev/null 2>&1 \
                    && { echo "[OK] launched Terminal.app window with live log"; launched_terminal=1; }
            fi
        fi
        ;;
    Linux)
        # Linux: walk a preference list of common terminal emulators.
        if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
            for term_bin in gnome-terminal pop-terminal-emulator konsole \
                            kitty alacritty foot terminator x-terminal-emulator xterm; do
                if command -v "$term_bin" >/dev/null; then
                    case "$term_bin" in
                        gnome-terminal|pop-terminal-emulator)
                            "$term_bin" --title="$TERM_TITLE" -- bash -c "$TAIL_CMD" >/dev/null 2>&1 &
                            ;;
                        konsole)
                            "$term_bin" -p "tabtitle=$TERM_TITLE" -e bash -c "$TAIL_CMD" >/dev/null 2>&1 &
                            ;;
                        kitty|alacritty|foot)
                            "$term_bin" -T "$TERM_TITLE" -e bash -c "$TAIL_CMD" >/dev/null 2>&1 &
                            ;;
                        terminator)
                            "$term_bin" -T "$TERM_TITLE" -x bash -c "$TAIL_CMD" >/dev/null 2>&1 &
                            ;;
                        x-terminal-emulator|xterm)
                            "$term_bin" -T "$TERM_TITLE" -e "$TAIL_CMD" >/dev/null 2>&1 &
                            ;;
                    esac
                    echo "[OK] launched $term_bin window with live log"
                    launched_terminal=1
                    break
                fi
            done
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # Windows (Git Bash / MSYS / Cygwin). wt.exe (Windows
        # Terminal) is the modern path; fall back to start cmd.
        # This branch is untested by the maintainers; patches welcome.
        if command -v wt.exe >/dev/null; then
            wt.exe new-tab --title "$TERM_TITLE" bash -c "$TAIL_CMD" >/dev/null 2>&1 &
            echo "[OK] launched Windows Terminal tab with live log"
            launched_terminal=1
        elif command -v cmd.exe >/dev/null; then
            cmd.exe /c start "" "bash" -c "$TAIL_CMD" >/dev/null 2>&1 &
            echo "[OK] launched cmd-spawned bash with live log"
            launched_terminal=1
        fi
        ;;
esac

if [ "$launched_terminal" -eq 0 ]; then
    echo "[INFO] no graphical terminal detected on this OS / session."
    echo "       Run this in your own terminal to follow the log:"
    echo "         tail -f $LOG"
    echo "       Or attach the tmux session:"
    echo "         tmux attach -t $SESSION   (Ctrl-b d to detach)"
fi
