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

# Date in YYYY · MM · DD format
DATE="$(date +%Y' · '%m' · '%d)"

# Build tag spans
TAG_HTML=""
if [[ -n "$TAGS" ]]; then
  IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
  for tag in "${TAG_ARRAY[@]}"; do
    tag="$(echo "$tag" | xargs)"  # trim whitespace
    TAG_HTML="$TAG_HTML      <span class=\"log-tag\">$tag</span>
"
  done
fi

# Create post directory and HTML file
mkdir -p "$POST_DIR"

cat > "$POST_DIR/post.html" << POSTEOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>$TITLE · justesen.xyz</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../../styles.css">
</head>
<body>

  <div class="page">

    <div class="page-header">
      <h1 class="title"><a href="../../index.html" style="color:inherit;text-decoration:none">justesen<span class="dot">.</span><span class="tld">xyz</span></a></h1>
      <nav class="page-nav">
        <a href="../../about.html">about</a>
        <a href="../../work.html">work</a>
        <a href="../../log.html" class="active">log</a>
        <a href="../../contact.html">contact</a>
      </nav>
    </div>

    <a href="../../log.html" class="post-back">← back to log</a>

    <div class="log-date">$DATE</div>
    <div class="log-title" style="font-size:15px;margin-bottom:4px;">$TITLE</div>
    <div class="log-tags" style="margin-bottom:0;">
$TAG_HTML    </div>

    <div class="post-body">

      <h2>Introduction</h2>

      <p>
        Start writing here.
      </p>

    </div>

    <footer class="footer" style="width:100%">&copy; 2026 justesen.xyz</footer>

  </div>

  <div class="lightbox-overlay" id="lightbox">
    <img id="lightbox-img" src="" alt="">
  </div>
  <div class="lightbox-hint">click or press esc to close</div>

  <script>
    const overlay = document.getElementById('lightbox');
    const lbImg = document.getElementById('lightbox-img');

    document.querySelectorAll('.post-body img').forEach(img => {
      img.addEventListener('click', () => {
        lbImg.src = img.src;
        lbImg.alt = img.alt;
        overlay.classList.add('active');
      });
    });

    overlay.addEventListener('click', () => {
      overlay.classList.remove('active');
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') overlay.classList.remove('active');
    });
  </script>

</body>
</html>
POSTEOF

# Build log entry tag spans (indented for log.html context)
LOG_TAG_HTML=""
if [[ -n "$TAGS" ]]; then
  IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
  for tag in "${TAG_ARRAY[@]}"; do
    tag="$(echo "$tag" | xargs)"
    LOG_TAG_HTML="$LOG_TAG_HTML        <span class=\"log-tag\">$tag</span>
"
  done
fi

# Build the new log entry
LOG_ENTRY="\\
    <div class=\"log-entry\">\\
      <div class=\"log-date\">$DATE</div>\\
      <div class=\"log-title\"><a href=\"posts/$SLUG/post.html\">$TITLE</a></div>\\
      <div class=\"log-excerpt\">\\
        TODO: Write a short excerpt.\\
      </div>\\
      <div class=\"log-tags\">\\
$(IFS=',' read -ra TAG_ARRAY <<< "$TAGS"; for tag in "${TAG_ARRAY[@]}"; do tag="$(echo "$tag" | xargs)"; echo "        <span class=\"log-tag\">$tag</span>\\"; done)
      </div>\\
    </div>"

# Insert after the section-label line in log.html
sed -i '' "/^    <div class=\"section-label\">log<\/div>$/a\\
\\
$LOG_ENTRY
" "$REPO_ROOT/log.html"

echo "Created posts/$SLUG/post.html"
echo "Added entry to log.html"
