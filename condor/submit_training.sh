#!/usr/bin/env bash

# Submits one training to the HTCondor batch system of the NAF.
#
# Usage (on a NAF login node, after 'source setup.sh'):
#   bash condor/submit_training.sh [options] <name> [-- <train.py arguments>]
#
# The current state of src/ (configs and uncommitted changes included) is copied into the job directory
# ${CONDOR_JOB_ROOT}/<name>, the job trains with this copy. Editing the configs after the submission
# therefore does not change queued or running jobs, so trainings can be prepared one after another:
# adapt the configs, submit, adapt the configs, submit, ...
#
# <name> names the job directory, the tensorboard run (-tn) and, unless --save-model-name is given,
# the checkpoints (-sn). A '-cs <suffix>' after '--' is appended to the checkpoint name as by train.py.
# The submission is refused if a checkpoint or another job directory already uses the checkpoint name.
#
# Options:
#   --gpus N       number of GPUs, 0 trains on CPU (default: 1)
#   --cpus N       number of CPU cores (default: 4)
#   --memory N     memory in GB (default: 16)
#   --runtime N    maximum runtime in hours (default: 24)
#   --save-model-name NAME
#                  name of the checkpoints instead of <name>
#   --save-cache   write the preprocessed data to the cache. By default jobs pass -ns to train.py, since
#                  parallel jobs would write the same cache files at the same time. Fill the cache once
#                  before (interactive training or one job with --save-cache), then submit the others.
#   --force        submit even if the checkpoint name is already used by a checkpoint or another job
#   --dry-run      create the job directory and submit file, but do not submit
#   -h, --help     show this help
#
# Everything after '--' is passed on to train.py, e.g. '-- -rc' to rebuild the cache or
# '-- -o optimizer_config.lr=1e-3' to override config values without editing the configs.
#
# CONDOR_JOB_ROOT (optional, e.g. in config.sh): directory of the job directories,
# defaults to ${STORE_DIR}/condor.

set -euo pipefail

usage() {
    # prints the comment block at the top of this file
    sed -n '3,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

condor_quote() {
    # Quotes one argument for the 'arguments' command of a submit file (new syntax: inside the
    # double quoted list, single quotes group an argument, quote characters are escaped by doubling).
    #
    # Args:
    #   $1 (str): argument to quote
    # quote characters in variables, backslashes in the replacement behave differently between bash versions
    local arg="$1" sq="'" dq='"'
    arg="${arg//${sq}/${sq}${sq}}"
    arg="${arg//${dq}/${dq}${dq}}"
    printf "'%s'" "${arg}"
}

# --- arguments
gpus=1
cpus=4
memory=16
runtime=24
save_cache=0
force=0
dry_run=0
name=""
model_name=""
train_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --memory) memory="$2"; shift 2 ;;
        --runtime) runtime="$2"; shift 2 ;;
        --save-model-name) model_name="$2"; shift 2 ;;
        --save-cache) save_cache=1; shift ;;
        --force) force=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; train_args=("$@"); break ;;
        -*) echo "[ERROR] Unknown option $1, see --help" >&2; exit 1 ;;
        *)
            if [[ -n "${name}" ]]; then
                echo "[ERROR] Got a second name '$1', train.py arguments go after '--'" >&2
                exit 1
            fi
            name="$1"
            shift
            ;;
    esac
done

for value in "${gpus}" "${cpus}" "${memory}" "${runtime}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --gpus, --cpus, --memory and --runtime expect integers, got '${value}'" >&2
        exit 1
    fi
done

if [[ -z "${name}" ]]; then
    echo "[ERROR] No name given, see --help" >&2
    exit 1
fi
model_name="${model_name:-${name}}"
for value in "${name}" "${model_name}"; do
    if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[ERROR] Names may only contain letters, digits, '.', '_' and '-', got '${value}'" >&2
        exit 1
    fi
done

# the checkpoint name is set by this script (-sn), a '-cs' suffix is appended by train.py
model_name_base="${model_name}"
i=0
while [[ ${i} -lt ${#train_args[@]} ]]; do
    arg="${train_args[${i}]}"
    case "${arg}" in
        -sn|--save_model_name|--save_model_name=*)
            echo "[ERROR] Set the checkpoint name with --save-model-name before '--', not with ${arg}" >&2
            exit 1
            ;;
        -cs|--checkpoint_suffix)
            i=$((i + 1))
            model_name="${model_name}-${train_args[${i}]:-}"
            ;;
        --checkpoint_suffix=*)
            model_name="${model_name}-${arg#*=}"
            ;;
    esac
    i=$((i + 1))
done

# --- environment
if [[ -z "${SETUP_COMPLETE:-}" ]]; then
    echo "[ERROR] Run 'source setup.sh' first" >&2
    exit 1
fi
# the job sets up the naf environment, so it must have been created on the naf
if [[ "${ML_SYSTEM}" != "naf" ]]; then
    echo "[ERROR] HTCondor jobs run on the NAF, but the environment is set up for '${ML_SYSTEM}'" >&2
    exit 1
fi
if [[ "${dry_run}" == "0" ]] && ! command -v condor_submit >/dev/null 2>&1; then
    echo "[ERROR] condor_submit not found, submit from a NAF login node" >&2
    exit 1
fi

job_root="${CONDOR_JOB_ROOT:-${STORE_DIR}/condor}"
job_dir="${job_root}/${name}"
if [[ -e "${job_dir}" ]]; then
    echo "[ERROR] ${job_dir} already exists, choose another name or remove it" >&2
    exit 1
fi

# --- job directory with a copy of the code
mkdir -p "${job_dir}"
# on errors below the incomplete job directory is removed again
trap 'rm -rf "${job_dir}"' ERR

rsync -a --exclude "__pycache__" --exclude "*.egg-info" "${DNN_ROOT}/src/" "${job_dir}/src/"
cp "${DNN_ROOT}/condor/run_training.sh" "${job_dir}/run_training.sh"
chmod +x "${job_dir}/run_training.sh"

# state of the repository, to know later which code a training used (the copy also contains it)
{
    echo "commit: $(git -C "${DNN_ROOT}" rev-parse HEAD)"
    echo "branch: $(git -C "${DNN_ROOT}" rev-parse --abbrev-ref HEAD)"
    echo "status:"
    git -C "${DNN_ROOT}" status --short
} > "${job_dir}/git_state.txt"
git -C "${DNN_ROOT}" diff HEAD > "${job_dir}/git_diff.patch"

# --- checkpoint name, recorded to find collisions with later submissions
echo "${model_name}" > "${job_dir}/save_model_name.txt"

if [[ "${force}" == "0" ]]; then
    conflicts=()
    for checkpoint in "${MODELS_DIR}/${model_name}"_fold*.pt; do
        if [[ -e "${checkpoint}" ]]; then
            conflicts+=("checkpoint ${checkpoint}")
        fi
    done
    for other in "${job_root}"/*/save_model_name.txt; do
        if [[ -e "${other}" && "${other}" != "${job_dir}/save_model_name.txt" && "$(cat "${other}")" == "${model_name}" ]]; then
            conflicts+=("job $(dirname "${other}")")
        fi
    done
    if [[ ${#conflicts[@]} -gt 0 ]]; then
        echo "[ERROR] Checkpoint name '${model_name}' is already used by:" >&2
        printf '    %s\n' "${conflicts[@]}" >&2
        echo "        Choose another name (--save-model-name) or pass --force to overwrite" >&2
        rm -rf "${job_dir}"
        exit 1
    fi
fi

# --- submit file
# -sn without a possible -cs suffix, train.py appends it; it goes first, '-o' takes all values after it
train_args=("-sn" "${model_name_base}" ${train_args[@]+"${train_args[@]}"})
if [[ "${save_cache}" == "0" ]]; then
    train_args=("-ns" "${train_args[@]}")
fi

arguments=""
for arg in "${DNN_ROOT}" "${job_dir}" "${name}" ${train_args[@]+"${train_args[@]}"}; do
    arguments+="$(condor_quote "${arg}") "
done

{
    echo "# created by condor/submit_training.sh"
    echo "executable      = ${job_dir}/run_training.sh"
    echo "arguments       = \"${arguments% }\""
    echo "initialdir      = ${job_dir}"
    echo "log             = ${job_dir}/job.log"
    echo "output          = ${job_dir}/job.out"
    echo "error           = ${job_dir}/job.err"
    echo "getenv          = false"
    echo "environment     = \"REQUESTED_GPUS=${gpus}\""
    echo "request_cpus    = ${cpus}"
    echo "request_memory  = ${memory} GB"
    if [[ "${gpus}" -gt 0 ]]; then
        echo "request_gpus    = ${gpus}"
    fi
    # NAF specific: maximum runtime in seconds
    echo "+RequestRuntime = $((runtime * 3600))"
    echo "queue"
} > "${job_dir}/job.sub"

trap - ERR

echo "[INFO] Job directory: ${job_dir}"
echo "[INFO] Checkpoint name: ${model_name}, tensorboard name: ${name}"
echo "[INFO] Request: ${gpus} GPU, ${cpus} CPU, ${memory} GB, ${runtime} h; train.py arguments: ${train_args[*]+${train_args[*]}}"

if [[ "${dry_run}" == "1" ]]; then
    echo "[INFO] Dry run, submit later with: condor_submit ${job_dir}/job.sub"
    exit 0
fi
condor_submit "${job_dir}/job.sub"
