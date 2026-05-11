#!/usr/bin/env bash
# run_fragment_library_local.sh
# ─────────────────────────────
# Runs the fragment library generation step locally (no SLURM).
#
# Usage:
#   bash scripts/local/run_fragment_library_local.sh [INPUT_JSON] [OPTIONS...]
#
# Arguments:
#   INPUT_JSON  Path to a JSON config file (default: examples/inputs/in_local.json)
#   OPTIONS...  Any extra flags forwarded directly to create_fragment_library.py
#
# Examples:
#   bash scripts/local/run_fragment_library_local.sh
#   bash scripts/local/run_fragment_library_local.sh my_project/in.json --cpus 16
#
# The script automatically sets --riff_diff_dir and --jobstarter Local.
# Make sure the correct conda environment (with protflow installed) is active.
set -euo pipefail

# ── Locate repo root (two levels up from this script) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFF_DIFF_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_JSON="${1:-${RIFF_DIFF_DIR}/examples/inputs/in_local.json}"
shift 2>/dev/null || true   # consume first arg if present; ignore error if no args

if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "ERROR: Input JSON not found at: ${INPUT_JSON}"
    echo "Usage: $0 [INPUT_JSON] [extra options...]"
    exit 1
fi

echo "================================================="
echo " RiffDiff — Fragment Library Generation (Local)"
echo "================================================="
echo "  Repo dir  : ${RIFF_DIFF_DIR}"
echo "  Input JSON: ${INPUT_JSON}"
echo "  Extra args: $*"
echo "================================================="

python "${RIFF_DIFF_DIR}/create_fragment_library.py" \
    --riff_diff_dir "${RIFF_DIFF_DIR}" \
    --input_json    "${INPUT_JSON}" \
    --jobstarter    Local \
    "$@"

echo ""
echo "Fragment library generation complete."
echo "Output in: $(python -c "import json; d=json.load(open('${INPUT_JSON}')); print(d.get('working_dir','outputs'))")"
