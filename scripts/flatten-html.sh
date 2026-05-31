#!/usr/bin/env bash
# Convert Zola's `foo.html/index.html` directories into flat `foo.html` files
# so the output tree matches Jekyll's permalink layout exactly, then rewrite
# every `foo.html/` URL (escaped or raw) in the built output to `foo.html`.
# Zola unconditionally appends a trailing slash to `page.permalink` and the
# theme uses that in href/link tags; without the rewrite, every link from the
# home page / feed / search index would point to a now-nonexistent slashed URL.
#
# Usage: flatten-html.sh <public-dir>
set -euo pipefail

root="${1:?usage: $0 <public-dir>}"

# Step 1: flatten foo.html/index.html -> foo.html
find "$root" -depth -type d -name '*.html' | while read -r d; do
  idx="$d/index.html"
  [ -f "$idx" ] || continue
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

# Step 2: rewrite ".html/" and ".html&#x2F;" -> ".html" in built output.
find "$root" -type f \( -name '*.html' -o -name '*.xml' -o -name '*.js' \) -print0 \
  | xargs -0 perl -pi -e 's{\.html(?:&\#x2F;|/)}{.html}g'
