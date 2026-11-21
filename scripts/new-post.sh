#!/usr/bin/env bash
set -euo pipefail

SLUG="${1:-}"
TITLE="${2:-}"
TAGS="${3:-}"

if [[ -z "$SLUG" ]]; then
  echo "Usage: $0 <slug> <title> [tags]"
  echo "  slug   — directory name under posts/ (required)"
  echo "  title  — post title (required)"
  echo "  tags   — comma-separated tags (optional)"
  exit 1
fi

if [[ -z "$TITLE" ]]; then
  echo "Error: title is required"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POST_DIR="$REPO_ROOT/posts/$SLUG"

if [[ -d "$POST_DIR" ]]; then
  echo "Error: $POST_DIR already exists"
  exit 1
fi

# Date in YYYY · MM · DD format (log.html) and ISO format (front matter)
DATE="$(date +%Y' · '%m' · '%d)"
ISO_DATE="$(date +%Y-%m-%d)"

YAML_TITLE="${TITLE//\\/\\\\}"
YAML_TITLE="${YAML_TITLE//\"/\\\"}"

# Front matter tag list "a, b" -> [a, b] and the log.html tag spans
FM_TAGS=""
TAG_SPANS=""
if [[ -n "$TAGS" ]]; then
  IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
  for tag in "${TAG_ARRAY[@]}"; do
    tag="$(echo "$tag" | xargs)"  # trim whitespace
    FM_TAGS="${FM_TAGS:+$FM_TAGS, }$tag"
    TAG_SPANS+="        <span class=\"log-tag\">$tag</span>"$'\n'
  done
fi

# Create post directory and HTML file (header, footer, lightbox come from _layouts/post.html)
mkdir -p "$POST_DIR"

cat > "$POST_DIR/post.html" << POSTEOF
---
layout: post
title: "$YAML_TITLE"
date: $ISO_DATE
nav: log
tags: [$FM_TAGS]
---

      <h2>Introduction</h2>

      <p>
        Start writing here.
      </p>
POSTEOF

LOG_ENTRY="
    <div class=\"log-entry\">
      <div class=\"log-date\">$DATE</div>
      <div class=\"log-title\"><a href=\"posts/$SLUG/post.html\">$TITLE</a></div>
      <div class=\"log-excerpt\">
        TODO: Write a short excerpt.
      </div>
      <div class=\"log-tags\">
${TAG_SPANS}      </div>
    </div>"

# Insert the entry after the section-label line in log.html
LOG="$REPO_ROOT/log.html"
LABEL='    <div class="section-label">log</div>'
TMP="$(mktemp)"
while IFS= read -r line || [[ -n "$line" ]]; do
  printf '%s\n' "$line"
  if [[ "$line" == "$LABEL" ]]; then
    printf '%s\n' "$LOG_ENTRY"
  fi
done < "$LOG" > "$TMP"
cat "$TMP" > "$LOG" && rm "$TMP"

echo "Created posts/$SLUG/post.html"
echo "Added entry to log.html"
