# Environment setup for pytorch_network_playground.
#
# Usage (must be sourced, not executed):
#   source setup.sh
#
# Steps:
#   1. load config.sh (created from config.sh.example on first run)
#   2. create the storage directories if CREATE_DIR=1
#   3. activate the virtualenv ${VENV_ROOT}/${ML_ENV}

# this file modifies the calling shell, so it only works when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[ERROR] setup.sh must be sourced: 'source setup.sh'" >&2
    exit 1
fi


load_config() {
    # Sources config.sh next to this file; creates it from config.sh.example on first run.
    #
    # Returns:
    #   0 on success, 1 if config.sh was just created and still needs editing.
    local config="${DNN_ROOT}/config.sh"

    if [[ ! -f "${config}" ]]; then
        cp "${DNN_ROOT}/config.sh.example" "${config}"
        echo "[INFO] Created ${config} from template. Adapt it, then run 'source setup.sh' again."
        return 1
    fi

    source "${config}"

    # every variable below is needed by setup.sh or the python code
    local var
    for var in STORE_DIR CACHE_DIR PICTURE_DIR MODELS_DIR TENSORBOARD_DIR EVALUATION_DIR; do
        # ${!var} expands the variable whose name is stored in var
        if [[ -z "${!var}" ]]; then
            echo "[ERROR] ${var} is not set in config.sh" >&2
            return 1
        fi
    done
    return 0
}


create_dirs() {
    # Creates the storage directories when CREATE_DIR=1 in config.sh.
    #
    # Returns:
    #   0 on success or when skipped, 1 if a directory could not be created.
    if [[ "${CREATE_DIR}" != "1" ]]; then
        return 0
    fi

    local dir
    for dir in "${CACHE_DIR}" "${PICTURE_DIR}" "${MODELS_DIR}" "${TENSORBOARD_DIR}" "${EVALUATION_DIR}"; do
        if ! mkdir -p "${dir}"; then
            echo "[ERROR] Failed to create ${dir}" >&2
            return 1
        fi
    done
    echo "[INFO] Storage directories ready under ${STORE_DIR}"
    return 0
}


activate_venv() {
    # Activates a plain 'python -m venv' environment at ${VENV_ROOT}/${ML_ENV}.
    #
    # Returns:
    #   0 if the environment is active, 1 otherwise.
    if [[ -z "${VENV_ROOT}" || -z "${ML_ENV}" ]]; then
        echo "[ERROR] VENV_ROOT and ML_ENV must be set in config.sh" >&2
        return 1
    fi

    local venv_dir="${VENV_ROOT}/${ML_ENV}"
    if [[ ! -f "${venv_dir}/bin/activate" ]]; then
        echo "[ERROR] No virtualenv found at ${venv_dir}, see README for installation" >&2
        return 1
    fi

    # activate without prompt change, the prompt is set in finish_setup
    VIRTUAL_ENV_DISABLE_PROMPT=1 source "${venv_dir}/bin/activate" || return 1

    # the pip packages ship their own CUDA libraries (cudnn, nccl), make them visible to the loader
    # (needed when the system CUDA installation does not match the torch wheel)
    local site_packages
    site_packages="$(ls -d "${venv_dir}"/lib/python3.*/site-packages 2>/dev/null | head -n 1)"
    if [[ -n "${site_packages}" ]]; then
        local cuda_libs="${site_packages}/nvidia/cudnn/lib:${site_packages}/nvidia/nccl/lib"
        # ${VAR:+:${VAR}} appends ":old value" only if the old value was non-empty,
        # so no trailing ':' (which would mean "search the current directory") is produced
        export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    return 0
}


finish_setup() {
    # Marks the setup as complete and shows the environment name in the prompt.
    export SETUP_COMPLETE=1
    # ${VIRTUAL_ENV##*/} = last path component = name of the venv
    export PS1="[${VIRTUAL_ENV##*/}] ${PS1}"
    echo "Ready to go"
}


setup() {
    if [[ -n "${SETUP_COMPLETE}" ]]; then
        echo "Setup already completed. Skipping."
        return 0
    fi

    # directory of this file = repository root
    DNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DNN_ROOT

    # config.sh.example derives all personal paths from the system user name
    if [[ -z "${USER}" ]]; then
        echo "[ERROR] \$USER is not set, cannot derive paths" >&2
        return 1
    fi

    load_config || return 1
    create_dirs || return 1

    # debugging helpers pdb_on / pdb_off
    source "${DNN_ROOT}/.setup/pdb.sh"

    activate_venv || return 1
    finish_setup
}

setup
