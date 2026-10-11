#!/usr/bin/env bash
# Publish this folder as a GitHub repository, using YOUR git/GitHub login on this computer.
#   bash publish_to_github.sh <github-username> [repo-name] [--private]
set -euo pipefail
USER_NAME=${1:?usage: bash publish_to_github.sh <github-username> [repo-name] [--private]}
REPO=${2:-ininauto}
VIS=${3:---public}
cd "$(dirname "$0")"
[ -d .git ] || git init -q -b main
git add -A
git commit -q -m "InInAuto 0.3: action-to-number in-in solver" || echo "(nothing new to commit)"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$USER_NAME/$REPO" "$VIS" --source=. --remote=origin --push
else
  URL="https://github.com/$USER_NAME/$REPO.git"
  git remote add origin "$URL" 2>/dev/null || git remote set-url origin "$URL"
  echo "GitHub CLI not available/logged in."
  echo "1) Create an EMPTY repository named '$REPO' at https://github.com/new (no README, no licence)."
  read -r -p "2) Press Enter when it exists to push... " _
  git push -u origin main
fi
echo
echo "Done.  Install it anywhere with:"
echo "    pip install git+https://github.com/$USER_NAME/$REPO.git"
echo "and use it with:  import ininauto"
