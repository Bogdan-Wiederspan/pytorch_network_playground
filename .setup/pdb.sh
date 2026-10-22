# Post-mortem debugging helpers, sourced by setup.sh.
#
# pdb_on [pdb|ipdb|embed]   drop into a debugger when an uncaught exception occurs
# pdb_off                   disable again
#
# Works by putting this directory (containing sitecustomize.py) on PYTHONPATH;
# sitecustomize.py installs a sys.excepthook that reads PDB_DEBUGGER_MODE.

# absolute path of this directory, independent of the current working directory
PDB_HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PDB_HOOK_DIR

pdb_on() {
    # Enables the post-mortem hook.
    #
    # Args:
    #   $1 (str, optional): debugger mode, one of pdb, ipdb, embed. Defaults to embed.

    # small recap switch cases:
    # bash switch statement case VALUE in  PATTERN) commands ;; esac
    # VALUE is what is checked, pattern what is looked for
    # ;; ends case block
    # esac, is case backwards

    # set variable to change mode of debugger
    local mode="${1:-embed}"
    case "${mode}" in
        pdb|ipdb|embed) ;;
        *)
            echo "Usage: pdb_on [pdb|ipdb|embed]" >&2
            return 1
            ;;
    esac
    export PDB_DEBUGGER_MODE="${mode}"

    # first: check if PDB_HOOK_DIR already exist in PYTHONPATH, * is any token, when found break
    # second: *) match everything (is basically else)
    case ":${PYTHONPATH:-}:" in
        *":${PDB_HOOK_DIR}:"*)
            echo "pdb post-mortem hook already enabled (${mode})"
            ;;
        *)
            export PYTHONPATH="${PDB_HOOK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
            echo "pdb post-mortem hook: ON (${mode})"
            ;;
    esac
}

pdb_off() {
    # Disables the post-mortem hook by removing PDB_HOOK_DIR from PYTHONPATH.
    local entry
    local kept=()
    # split on ':' and drop the hook dir; IFS only changed for the read
    IFS=':' read -r -a _entries <<< "${PYTHONPATH:-}"
    for entry in "${_entries[@]}"; do
        [[ "${entry}" == "${PDB_HOOK_DIR}" ]] || kept+=("${entry}")
    done
    unset _entries

    if [[ ${#kept[@]} -gt 0 ]]; then
        export PYTHONPATH="$(IFS=':'; echo "${kept[*]}")"
    else
        unset PYTHONPATH
    fi
    unset PDB_DEBUGGER_MODE
    echo "pdb post-mortem hook: OFF"
}
