# Detects which DESY system we run on, sourced by setup.sh and install_env.sh.
#
# detect_host [naf|maxwell]   sets and exports ML_SYSTEM
#
# Without an argument the hostname decides: naf-* -> naf, max-* -> maxwell.
# An argument overrides the detection, e.g. 'source setup.sh maxwell'.

detect_host() {
    # Sets ML_SYSTEM from the argument or the hostname.
    #
    # Args:
    #   $1 (str, optional): naf or maxwell. Defaults to hostname based detection.
    #
    # Returns:
    #   0 on success, 1 if the system is unknown.
    local requested="${1:-}"

    if [[ -n "${requested}" ]]; then
        case "${requested}" in
            naf|maxwell) ML_SYSTEM="${requested}" ;;
            *)
                echo "[ERROR] Unknown system '${requested}', expected 'naf' or 'maxwell'" >&2
                return 1
                ;;
        esac
    else
        case "$(hostname)" in
            naf-*) ML_SYSTEM="naf" ;;
            max-*) ML_SYSTEM="maxwell" ;;
            *)
                echo "[ERROR] Cannot detect system from hostname '$(hostname)', pass it explicitly: naf or maxwell" >&2
                return 1
                ;;
        esac
    fi

    export ML_SYSTEM
    return 0
}
