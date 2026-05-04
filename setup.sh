
#!/bin/bash

activate_venv(){
    _path=${VENV_ROOT}/${ML_ENV}/bin/
    # activate but without prompt change
    VIRTUAL_ENV_DISABLE_PROMPT=1 source ${_path}/activate
    # set currently VENV as most important
    #export PATH="${_path}:${PATH}";
}

pyenv_activate ()
{
    # bootstrap pyenv to be activate for current bash session
    echo "Bootstrap Pyenv"
    export PATH="${PYENV_ROOT}/bin/:${PATH}";
    eval "$(pyenv init --path)";
    eval "$(pyenv init -)";
    eval "$(pyenv virtualenv-init -)"
}

setup_env() {
    # Setup Virtualenv and mount given environment in config

    # check if inside virtualenv -> if inside sys prefix changes
    # Source - https://stackoverflow.com/a/15454916
    local IN_VENV=$(python -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')
    # -n is true if NON-ZERO, -z true if EMPTY
    if [ "${IN_VENV}" = "1" ]; then
        echo "Already inside virtualenv"
        return 4
    fi

    # activate pyenv or cf setup
    if [ "${VENV_MODE}" = "pyenv" ]; then
        if [ -n "${PYENV_ROOT}" ] && [ -n "${ML_ENV}" ]; then
            # check if pyenv exists as command, if not activate it
            # command return 0 if exist, else another number
            # >/dev etc. just silence the output
            if ! command -v pyenv >/dev/null 2>&1; then
                pyenv_activate
            fi
            pyenv activate "${ML_ENV}"
            echo "activate virtualenv ${ML_ENV}"
        else
            echo "can' activate pyenv due too wrong PYEV_ROOT or not set ML_ENV"
            return 1
        fi
    # for bachelor and master students that use columnflow sandboxes
    elif [ "${VENV_MODE}" = "cf" ]; then
        echo "Sourcing cf sandbox"

        if [ -n "${CF_SANDBOX}" ]; then
            if command -v cf_sandbox >/dev/null 2>&1; then
                cf_sandbox ${CF_SANDBOX}
            else
                echo "Source your columnflow setup!"
                return 2
            fi
        else
            echo "CF_SANDBOX is not a valid columnflow sandbox - check if sandbox exist"
            return 2
        fi

    elif [ "${VENV_MODE}" = "venv" ]; then
        echo "Sourcing VENV Sandbox"
        activate_venv
    else
        echo "VENV_MODE can only be cf or pyenv, but got ${VENV_MODE}"
        return 3
    fi
    return 0
}

setup() {
    # Prevent double execution, when SETUP_COMPLETE is set (at end of script) don't run setup twice.
    if [ -n "$SETUP_COMPLETE" ]; then
        echo "Setup already completed. Skipping."
        return 0
    fi

    # Get the directory where this script is located and source config located there
    # BASH_SOURCE[0] = path of current script, get dir and resolve absolute path
    local LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$LOCAL_DIR/config.sh"

    # run env and check status if run correctly change python path and venv marker
    setup_env
    SETUP_DONE=$?

    if [ "${SETUP_DONE}" -eq 0 ]; then
        echo "Ready to go"
        export SETUP_COMPLETE=1
        # extend PS1 if environment is set properly
        # Show only the last part of the virtualenv path (env name) in the prompt
        export PS1="[${VIRTUAL_ENV##*/}] ${PS1}"
        # include source of project as root for python
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    fi
    return 0
}


setup "$@"
