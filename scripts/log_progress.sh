#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACKER_FILE="${ROOT_DIR}/README_PROGRESS.md"

usage() {
  cat <<'EOF'
Usage:
  scripts/log_progress.sh "Area" "Change" "Result" "Output paths"

Example:
  scripts/log_progress.sh \
    "Pipeline" \
    "Edge-artifact suppression + contour cleanup tuned" \
    "Success" \
    "runs/pipeline/id_xh32sn_final.jpg"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 4 ]]; then
  echo "Error: missing required arguments."
  usage
  exit 1
fi

AREA="$1"
CHANGE="$2"
RESULT="$3"
OUTPUT_PATHS="$4"

if [[ ! -f "${TRACKER_FILE}" ]]; then
  echo "Error: tracker file not found at ${TRACKER_FILE}"
  exit 1
fi

STAMP="$(date '+%Y-%m-%d %H:%M')"
ENTRY="- ${STAMP} | ${AREA} | ${CHANGE} | ${RESULT} | ${OUTPUT_PATHS}"

echo "${ENTRY}" >> "${TRACKER_FILE}"
echo "Appended to ${TRACKER_FILE}:"
echo "${ENTRY}"
