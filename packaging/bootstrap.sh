#!/bin/sh
# LAI one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/cufelix/lai/main/packaging/bootstrap.sh | sh
#
# Fetches the source, then hands over to packaging/install.sh, which installs
# system packages, builds the virtualenv, and runs the setup wizard.
#
# Environment:
#   LAI_DIR     where to install         (default: ~/.local/share/lai)
#   LAI_REPO    git URL to clone from    (default: the one baked in below)
#   LAI_REF     branch or tag            (default: main)
#
# Flags are passed through to install.sh, e.g.:
#   curl -fsSL … | sh -s -- --no-setup
#
# Deliberately POSIX sh: this is the one file that must run before anything is
# known about the machine, so it cannot assume bash.
set -eu

REPO="${LAI_REPO:-https://github.com/cufelix/lai.git}"
REF="${LAI_REF:-main}"
DIR="${LAI_DIR:-$HOME/.local/share/lai}"

if [ -t 1 ]; then
  BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m')
  GREEN=$(printf '\033[32m'); YELLOW=$(printf '\033[33m')
  RED=$(printf '\033[31m'); OFF=$(printf '\033[0m')
else
  BOLD=''; DIM=''; GREEN=''; YELLOW=''; RED=''; OFF=''
fi

say()  { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$RED" "$*" "$OFF" >&2; exit 1; }

printf '\n%sLAI%s — a native agent for your Linux desktop\n' "$BOLD" "$OFF"
printf '%sinstalling to %s%s\n' "$DIM" "$DIR" "$OFF"

# -- prerequisites --------------------------------------------------------

say "Checking prerequisites"

[ "$(uname -s)" = "Linux" ] || die "LAI drives a Linux desktop; this is $(uname -s)."
ok "Linux"

command -v git >/dev/null 2>&1 || die "git is required. Install it and run this again."
command -v python3 >/dev/null 2>&1 || die "python3 is required. Install it and run this again."
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null \
  || die "Python 3.10+ is required (found $(python3 -V 2>&1))."
ok "git, $(python3 -V 2>&1)"

case "${XDG_SESSION_TYPE:-}" in
  x11) ok "X11 session" ;;
  wayland) warn "Wayland session — LAI needs X11; log out and pick the Xorg session" ;;
  *) warn "session type '${XDG_SESSION_TYPE:-unknown}' — LAI expects x11" ;;
esac

# -- fetch ----------------------------------------------------------------

say "Fetching LAI"

case "$REPO" in
  *OWNER/lai*)
    die "This script has no repository URL baked in yet.
    Set one explicitly:
      LAI_REPO=https://github.com/you/lai.git sh -c \"\$(curl -fsSL …)\"" ;;
esac

if [ -d "$DIR/.git" ]; then
  # Re-running the installer must update rather than fail.
  ok "already cloned — updating"
  git -C "$DIR" fetch --quiet origin "$REF" || die "could not fetch $REF from $REPO"
  git -C "$DIR" checkout --quiet "$REF" 2>/dev/null || true
  git -C "$DIR" reset --hard --quiet "origin/$REF" 2>/dev/null \
    || git -C "$DIR" reset --hard --quiet "$REF"
else
  if [ -e "$DIR" ]; then
    die "$DIR exists but is not a git checkout. Move it, or set LAI_DIR."
  fi
  mkdir -p "$(dirname "$DIR")"
  git clone --quiet --depth 1 --branch "$REF" "$REPO" "$DIR" \
    || die "could not clone $REPO (branch $REF)"
fi
ok "source in $DIR"

[ -x "$DIR/packaging/install.sh" ] || chmod +x "$DIR/packaging/install.sh" 2>/dev/null || true
[ -f "$DIR/packaging/install.sh" ] || die "this checkout has no packaging/install.sh"

# -- hand over ------------------------------------------------------------

say "Running the installer"

# Piped into `sh`, our stdin *is* the script, so the installer and the setup
# wizard would read EOF instead of the human and silently take every default.
# Re-attaching the terminal is what makes `curl … | sh` behave like running the
# script directly. Without a tty (CI, a provisioning script) the wizard already
# falls back to defaults, so there is nothing to fix.
if [ ! -t 0 ]; then
  # `[ -r /dev/tty ]` is not the test: the device node can be readable while
  # the process has no controlling terminal at all, and the redirect then
  # fails hard. Attempting it in a subshell is the only reliable probe.
  if (exec < /dev/tty) 2>/dev/null; then
    exec < /dev/tty
  else
    warn "no terminal available — the setup wizard will take every default"
    warn "for the guided version, run: $DIR/packaging/install.sh"
  fi
fi

cd "$DIR"
exec bash packaging/install.sh "$@"
