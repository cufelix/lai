#!/usr/bin/env bash
# LAI installer — one command from a bare machine to a working agent.
#
#   ./packaging/install.sh
#
# It installs system packages, builds the virtualenv, puts `lai` on your PATH,
# and then hands over to `lai setup`, which is where the actual configuration
# happens. Everything it changes on your system is printed before it runs.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$RED" "$*" "$OFF"; exit 1; }

SKIP_SETUP=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --no-setup) SKIP_SETUP=1 ;;
    --yes|-y)   ASSUME_YES=1 ;;
    --help|-h)
       sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      echo "Options: --no-setup (stop before the wizard)  --yes (accept defaults)"
      exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

# -- 1. can this machine run LAI at all? ---------------------------------

say "Checking this machine"

[ "$(uname -s)" = "Linux" ] || die "LAI drives a Linux desktop; this is $(uname -s)."

command -v python3 >/dev/null 2>&1 || die "python3 is not installed."
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
  || die "Python 3.10+ required, found $PY_VERSION."
ok "python $PY_VERSION"

if [ "${XDG_SESSION_TYPE:-}" = "x11" ]; then
  ok "X11 session"
elif [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
  warn "this is a Wayland session — LAI needs X11"
  warn "log out, then pick the Xorg / X11 session at the login screen"
else
  warn "session type '${XDG_SESSION_TYPE:-unknown}' — LAI expects x11"
fi

# -- 2. system packages ---------------------------------------------------

say "System packages"

# python3-gi (AT-SPI, GTK) has no PyPI equivalent, so it must come from the
# distribution — which is also why the virtualenv needs --system-site-packages.
if command -v apt-get >/dev/null 2>&1; then
  PACKAGES=(xdotool python3-gi gir1.2-atspi-2.0 gir1.2-gtk-3.0 python3-venv)
  MISSING=()
  for pkg in "${PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
  done
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  ${DIM}sudo apt-get install -y ${MISSING[*]}${OFF}"
    sudo apt-get install -y "${MISSING[@]}"
    ok "installed ${#MISSING[@]} package(s)"
  else
    ok "all present"
  fi
elif command -v dnf >/dev/null 2>&1; then
  echo "  ${DIM}sudo dnf install -y xdotool python3-gobject gtk3 at-spi2-core${OFF}"
  sudo dnf install -y xdotool python3-gobject gtk3 at-spi2-core
  ok "installed"
elif command -v pacman >/dev/null 2>&1; then
  echo "  ${DIM}sudo pacman -S --needed --noconfirm xdotool python-gobject gtk3 at-spi2-core${OFF}"
  sudo pacman -S --needed --noconfirm xdotool python-gobject gtk3 at-spi2-core
  ok "installed"
else
  warn "unknown package manager — install these yourself:"
  warn "  xdotool, python3-gi, gir1.2-atspi-2.0, gir1.2-gtk-3.0"
fi

# -- 3. the virtualenv ----------------------------------------------------

say "Python environment"
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
  ok "created .venv (with system site packages, for python3-gi)"
else
  ok ".venv already exists"
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[mcp,tui]"
ok "installed LAI and its dependencies"

# -- 4. accessibility -----------------------------------------------------

say "Accessibility"
if command -v gsettings >/dev/null 2>&1; then
  current="$(gsettings get org.gnome.desktop.interface toolkit-accessibility 2>/dev/null || echo unknown)"
  if [ "$current" != "true" ]; then
    gsettings set org.gnome.desktop.interface toolkit-accessibility true 2>/dev/null \
      && ok "toolkit-accessibility: $current -> true" \
      || warn "could not set toolkit-accessibility (not a GNOME-based desktop?)"
    warn "applications already running must be restarted before they publish a tree"
  else
    ok "already enabled"
  fi
else
  warn "gsettings not found — enable accessibility in your desktop settings"
fi

# -- 5. put `lai` on PATH -------------------------------------------------

say "Command"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
ln -sf "$REPO_DIR/.venv/bin/lai" "$BIN_DIR/lai"
ok "linked $BIN_DIR/lai"

case ":$PATH:" in
  *":$BIN_DIR:"*) ok "$BIN_DIR is on your PATH" ;;
  *)
    warn "$BIN_DIR is not on your PATH — add this to your shell profile:"
    printf '      %sexport PATH="$HOME/.local/bin:$PATH"%s\n' "$DIM" "$OFF"
    ;;
esac

# -- 6. hand over to the wizard -------------------------------------------

if [ "$SKIP_SETUP" -eq 1 ]; then
  say "Done"
  echo "  Run ${BOLD}lai setup${OFF} to finish."
  exit 0
fi

say "Configuration"
echo "  ${DIM}handing over to \`lai setup\` — it will ask about your model backend${OFF}"
if [ "$ASSUME_YES" -eq 1 ]; then
  "$REPO_DIR/.venv/bin/lai" setup --yes
else
  "$REPO_DIR/.venv/bin/lai" setup
fi
