#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/external-drive [export-subdir]" >&2
  exit 2
fi

DEST_ROOT="$1"
EXPORT_NAME="${2:-Thesis_week05_week06_export}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${DEST_ROOT%/}/${EXPORT_NAME}"

if [[ ! -d "$DEST_ROOT" ]]; then
  echo "Destination root does not exist: $DEST_ROOT" >&2
  exit 2
fi

mkdir -p "$DEST"

PATHS=(
  "week05/data"
  "week05/results"
  "week05/results_npy_samples"
  "week05/fid_results"
  "week05/guidance_consistency_results"
  "week05/guidance_consistency_results_gradclip010"
  "week05/logs"
  "week05/scripts"
  "week05/slurm"
  "week05/WEEK05_CHECKPOINT_ARTIFACTS.md"
  "week06/report.md"
  "week06/PLAN.md"
  "week06/WEEK06_FAILURE_ANALYSIS.md"
  "week06/WEEK06_INITIAL_FINDINGS.md"
  "week06/WEEK06_WEEK05_OSM_MAP_REPORT.md"
  "week06/decoded"
  "week06/results"
  "week06/scripts"
  "week06/slurm"
  "src/cnr_trajectory/guidance"
  "tests/test_road_distance_guidance.py"
)

echo "Export destination: $DEST"
echo "Writing manifest..."
{
  echo "export_created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root: $REPO_ROOT"
  echo "git_head: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  echo
  echo "paths:"
  for rel in "${PATHS[@]}"; do
    if [[ -e "$REPO_ROOT/$rel" ]]; then
      size="$(du -sh "$REPO_ROOT/$rel" 2>/dev/null | awk '{print $1}')"
      echo "  - $rel ($size)"
    fi
  done
} > "$DEST/export_manifest.txt"

echo "Copying artifacts with rsync..."
for rel in "${PATHS[@]}"; do
  src="$REPO_ROOT/$rel"
  if [[ -e "$src" ]]; then
    mkdir -p "$DEST/$(dirname "$rel")"
    rsync -a --info=progress2 "$src" "$DEST/$(dirname "$rel")/"
  else
    echo "Skipping missing path: $rel"
  fi
done

echo "Done: $DEST"
