#!/usr/bin/env bash
# run_full_pipeline_local.sh
# ──────────────────────────
# Runs the complete RiffDiff pipeline locally:
#   Step 1: Fragment library generation (create_fragment_library.py)
#   Step 2: Structure generation       (structure_generation.py)
#
# Usage:
#   bash scripts/local/run_full_pipeline_local.sh INPUT_JSON [WORKING_DIR] [OPTIONS...]
#
# Arguments:
#   INPUT_JSON   Path to fragment library input JSON (required)
#   WORKING_DIR  Output directory (default: outputs/ relative to INPUT_JSON dir)
#   OPTIONS...   Extra flags forwarded to structure_generation.py
#
# Example:
#   bash scripts/local/run_full_pipeline_local.sh \
#       examples/inputs/in_local.json \
#       examples/outputs \
#       --max_cpus 8 \
#       --screen_num_rfdiffusions 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFF_DIFF_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 INPUT_JSON [WORKING_DIR] [structure_generation options...]"
    echo ""
    echo "  INPUT_JSON   Fragment library input JSON (required)"
    echo "  WORKING_DIR  Output directory (optional, default: outputs/)"
    echo ""
    echo "Example:"
    echo "  $0 examples/inputs/in_local.json examples/outputs --max_cpus 8"
    exit 1
fi

INPUT_JSON="$(realpath "${1}")"
shift

# Second arg is working dir if it doesn't start with '--'
if [[ $# -ge 1 && "${1}" != --* ]]; then
    WORKING_DIR="$(realpath "${1}")"
    shift
else
    # Read working_dir from JSON if present, else default to 'outputs'
    WORKING_DIR="$(python3 -c "
import json, os, sys
try:
    d = json.load(open('${INPUT_JSON}'))
    wd = d.get('working_dir', 'outputs')
    # resolve relative to JSON directory
    base = os.path.dirname('${INPUT_JSON}')
    print(os.path.realpath(os.path.join(base, wd)))
except Exception as e:
    print(os.path.realpath('outputs'))
")"
fi

DEFAULT_CPUS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

echo "==========================================================="
echo " RiffDiff — Full Pipeline (Local)"
echo "==========================================================="
echo "  Repo dir   : ${RIFF_DIFF_DIR}"
echo "  Input JSON : ${INPUT_JSON}"
echo "  Working dir: ${WORKING_DIR}"
echo "  CPU cores  : ${DEFAULT_CPUS}"
echo "  Extra args : $*"
echo "==========================================================="

# ── Step 1: Fragment library ────────────────────────────────────────────────
echo ""
echo "=== STEP 1/2: Fragment Library Generation ==="
python "${RIFF_DIFF_DIR}/create_fragment_library.py" \
    --riff_diff_dir "${RIFF_DIFF_DIR}" \
    --input_json    "${INPUT_JSON}" \
    --jobstarter    Local \
    --cpus          "${DEFAULT_CPUS}"

SELECTED_PATHS="${WORKING_DIR}/selected_paths.json"

if [[ ! -f "${SELECTED_PATHS}" ]]; then
    echo "ERROR: Expected fragment library output not found at: ${SELECTED_PATHS}"
    echo "Check the fragment library logs in ${WORKING_DIR}."
    exit 1
fi

echo ""
echo "Fragment library complete. Selected paths: ${SELECTED_PATHS}"

# ── Step 2: Structure generation ────────────────────────────────────────────
echo ""
echo "=== STEP 2/2: Structure Generation ==="
python "${RIFF_DIFF_DIR}/structure_generation.py" \
    --riff_diff_dir     "${RIFF_DIFF_DIR}" \
    --working_dir       "${WORKING_DIR}" \
    --screen_input_json "${SELECTED_PATHS}" \
    --jobstarter        Local \
    --max_cpus          "${DEFAULT_CPUS}" \
    --max_gpus          1 \
    "$@"

echo ""
echo "==========================================================="
echo " Full pipeline complete! Results in: ${WORKING_DIR}"
echo "==========================================================="
