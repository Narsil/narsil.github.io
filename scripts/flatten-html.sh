#!/usr/bin/env bash
# Convert Zola's `foo.html/index.html` directories into flat `foo.html` files
# so the output tree matches Jekyll's permalink layout exactly.
#
# Usage: flatten-html.sh <public-dir>
set -euo pipefail

root="${1:?usage: $0 <public-dir>}"

# Walk depth-first so nested .html dirs (none expected, but safe) flatten correctly.
find "$root" -depth -type d -name '*.html' | while read -r d; do
  idx="$d/index.html"
  [ -f "$idx" ] || continue
  # Refuse to flatten if the dir contains anything other than index.html.
  others=$(find "$d" -mindepth 1 -maxdepth 1 ! -name 'index.html' -print -quit)
  if [ -n "$others" ]; then
    echo "skip $d (unexpected extra entries)" >&2
    continue
  fi
  tmp="${d}.flat.tmp"
  mv "$idx" "$tmp"
  rmdir "$d"
  mv "$tmp" "$d"
done
