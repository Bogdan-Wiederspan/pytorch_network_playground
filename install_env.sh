#!/usr/bin/env bash
# Creates the virtualenv configured in config.sh and installs requirements.txt into it.
# Run once after creating config.sh:
#   ./install_venv.sh [python executable]
#
# The optional argument selects the python used to create the venv (default: python3),
# e.g. ./install_venv.sh python3.11

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${1:-python3}"

if [[ ! -f "${root}/config.sh" ]]; then
    echo "[ERROR] ${root}/config.sh not found, run 'source setup.sh' once to create it" >&2
    exit 1
fi
source "${root}/config.sh"

if [[ -z "${VENV_ROOT:-}" || -z "${ML_ENV:-}" ]]; then
    echo "[ERROR] VENV_ROOT and ML_ENV must be set in config.sh" >&2
    exit 1
fi

venv_dir="${VENV_ROOT}/${ML_ENV}"
if [[ -f "${venv_dir}/bin/activate" ]]; then
    echo "[ERROR] ${venv_dir} already exists, remove it first to reinstall" >&2
    exit 1
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[ERROR] ${python_bin} not found" >&2
    exit 1
fi
echo "[INFO] Creating ${venv_dir} with $("${python_bin}" --version)"

mkdir -p "${VENV_ROOT}"
"${python_bin}" -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
pip install --upgrade pip
pip install -r "${root}/requirements.txt"   # also installs this package in editable mode (-e .)

echo "[INFO] Done. Activate with 'source setup.sh'"
