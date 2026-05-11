#!/usr/bin/env bash
# run_structure_generation_local.sh
# ──────────────────────────────────
# Runs structure generation (screening → refinement → evaluation) locally.
# No SLURM required.
#
# Usage:
#   bash scripts/local/run_structure_generation_local.sh [OPTIONS...]
#
# Required (one of):
#   --screen_input_json PATH   Start from screening (output of fragment library step)
#   --ref_input_json    PATH   Start from refinement (skip screening)
#   --eval_input_json   PATH   Start from evaluation (skip screening + refinement)
#
# Common options:
#   --working_dir PATH         Output directory (required if not in --config)
#   --config      PATH         JSON config file with default argument values
#   --max_cpus    N            Parallel CPU subprocesses (default: all cores)
#   --max_gpus    N            Parallel GPU subprocesses (default: 1)
#   --skip_refinement          Only run screening, then stop
#   --skip_evaluation          Run screening + refinement, then stop
#
# Example (minimal single-GPU run):
#   bash scripts/local/run_structure_generation_local.sh \
#       --screen_input_json outputs/selected_paths.json \
#       --working_dir       outputs/structure_gen \
#       --max_cpus 8
#
# Example with a config file:
#   bash scripts/local/run_structure_generation_local.sh \
#       --config examples/inputs/structure_generation_local.json
#
set -euo pipefail

# ── Locate repo root (two levels up from this script) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFF_DIFF_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default max_cpus to number of available logical cores if not supplied by caller
DEFAULT_CPUS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

echo "==========================================================="
echo " RiffDiff — Structure Generation (Local)"
echo "==========================================================="
echo "  Repo dir : ${RIFF_DIFF_DIR}"
echo "  Args     : $*"
echo "==========================================================="

python "${RIFF_DIFF_DIR}/structure_generation.py" \
    --riff_diff_dir "${RIFF_DIFF_DIR}" \
    --jobstarter    Local \
    --max_cpus      "${DEFAULT_CPUS}" \
    --max_gpus      1 \
    "$@"

echo ""
echo "Structure generation complete."
