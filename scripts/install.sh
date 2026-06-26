#!/usr/bin/env bash
# Install deepiri-ollama-utils via curl:
#   curl -fsSL https://raw.githubusercontent.com/Team-Deepiri/deepiri-ollama-utils/main/scripts/install.sh | bash
set -euo pipefail

REPO="Team-Deepiri/deepiri-ollama-utils"
REPO_URL="https://github.com/${REPO}.git"
BRANCH="${DEEPIRI_OLLAMA_UTILS_BRANCH:-main}"
KEEP_DIR="${DEEPIRI_OLLAMA_UTILS_KEEP_DIR:-0}"
RELEASE_TAG="${DEEPIRI_OLLAMA_UTILS_RELEASE_TAG:-}"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Install deepiri-ollama-utils from a GitHub Release wheel (if available) or editable pip install.

Options:
  -h, --help        Show this help
  --dry-run         Print actions without installing
  --from-source     Force editable install from git checkout (skip release wheel)

Environment:
  DEEPIRI_OLLAMA_UTILS_SRC         Existing checkout
  DEEPIRI_OLLAMA_UTILS_BRANCH      Git branch (default: main)
  DEEPIRI_OLLAMA_UTILS_KEEP_DIR    Keep clone when set to 1
  DEEPIRI_OLLAMA_UTILS_RELEASE_TAG Pin release tag for wheel download (optional)

Requires: git, python3 (>=3.9); Ollama running locally for full verify
Verify:   deepiri-ollama-utils check
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

DRY_RUN=0
FROM_SOURCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --from-source) FROM_SOURCE=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for cmd in git python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd is required." >&2; exit 1; }
done

ROOT=""
CLEANUP=""
LOCAL_BIN="${HOME}/.local/bin"

if [[ -n "${DEEPIRI_OLLAMA_UTILS_SRC:-}" && -f "${DEEPIRI_OLLAMA_UTILS_SRC}/pyproject.toml" ]]; then
  ROOT="${DEEPIRI_OLLAMA_UTILS_SRC}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != bash ]] && [[ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  ROOT="$(mktemp -d)"
  [[ "$KEEP_DIR" != "1" ]] && CLEANUP="$ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would clone ${REPO_URL} to ${ROOT} (or download release wheel)"
    log "Would install deepiri-ollama-utils CLI to ${LOCAL_BIN}"
    exit 0
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
fi

[[ "$DRY_RUN" -eq 1 ]] && { log "Would install from ${ROOT}"; exit 0; }

trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT

VENV="${ROOT}/.venv"
log "Creating venv at ${VENV}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel -q

WHEEL_INSTALLED=0
if [[ "$FROM_SOURCE" -eq 0 ]] && command -v curl >/dev/null 2>&1; then
  TAG="$RELEASE_TAG"
  if [[ -z "$TAG" ]]; then
    TAG="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tag_name',''))" 2>/dev/null || true)"
  fi
  if [[ -n "$TAG" ]]; then
    ASSETS="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" 2>/dev/null || true)"
    WHEEL_URL="$(printf '%s' "$ASSETS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for a in data.get('assets', []):
    n = a.get('name', '')
    if n.endswith('.whl') or n.endswith('.zip'):
        print(a['browser_download_url'])
        break
" 2>/dev/null || true)"
    if [[ -n "$WHEEL_URL" ]]; then
      WHEEL_FILE="${ROOT}/$(basename "$WHEEL_URL")"
      log "Downloading release asset from ${TAG}"
      curl -fsSL -o "$WHEEL_FILE" "$WHEEL_URL"
      "$VENV/bin/pip" install "$WHEEL_FILE" -q
      WHEEL_INSTALLED=1
    fi
  fi
fi

if [[ "$WHEEL_INSTALLED" -eq 0 ]]; then
  log "Installing editable from source"
  cd "$ROOT"
  "$VENV/bin/pip" install -e . -q
fi

mkdir -p "$LOCAL_BIN"
ln -sf "$VENV/bin/deepiri-ollama-utils" "$LOCAL_BIN/deepiri-ollama-utils"
export PATH="${LOCAL_BIN}:${PATH}"

deepiri-ollama-utils check 2>/dev/null || warn "Ollama may not be running; CLI is installed"
echo ""
echo "Verify: deepiri-ollama-utils check"
if [[ ":$PATH:" != *":${LOCAL_BIN}:"* ]]; then
  echo "Add to your shell profile: export PATH=\"${LOCAL_BIN}:\$PATH\""
fi
