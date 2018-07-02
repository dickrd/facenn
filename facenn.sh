#!/usr/bin/env bash
set -o nounset
set -o errexit

# find script directory
setup_basedir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]
    do
        DIR="$( cd -P "$( dirname "$source" )" && pwd )"
        source="$(readlink "$source")"
        [[ "$source" != /* ]] && source="$DIR/$source"
    done
    basedir="$( cd -P "$( dirname "$source" )" && pwd )"
    PYTHONPATH="$basedir"
}

# run python script
main() {
    if [ "$#" -eq 1 ] && [ -r "$1" ]
    then
        echo "---- CONFIG DUMP ----"
        cat "$1"
        echo "---- END ----"
        source "$1"
        setup_basedir

        while [ "$repeat" -gt 0 ]
        do
            for arg in "${args[@]}"
            do
                python "$basedir/$module" ${arg}
            done
            repeat=$(("$repeat" - 1))
        done
    else
        case "$1" in
            gan)
                shift
                python "$basedir/model/gan_vgg.py" $@
                ;;
            *)
                echo "usage: $0 <config> | gan [options]"
                ;;
        esac
    fi
}

main $@
