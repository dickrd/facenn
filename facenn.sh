#!/usr/bin/env bash
set -o nounset
set -o errexit

basedir="./"
repeat_time=1

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
    if [ ! "$#" -eq 1 ] || [ ! -r "$1" ]
    then
        echo "must provide a config file!"
        exit -1
    fi
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
}

main $@
