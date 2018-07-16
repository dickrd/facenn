#!/usr/bin/env bash
set -o nounset
set -o errexit
trap "exit" INT

print_help() {
    echo "usage: facenn <config> | gan [options]"
}

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
    export PYTHONPATH="$basedir"
}

# run python script
main() {
    setup_basedir
    if [ "$#" -eq 1 ] && [ -r "$1" ]
    then
        source "$1"
        git --work-tree="$basedir" --git-dir="$basedir/.git" diff | tee -a "$logs"
        echo "---- CONFIG DUMP ----" | tee -a "$logs"
        cat "$1" | tee -a "$logs"
        echo "---- END ----" | tee -a "$logs"

        while [ "$repeat" -gt 0 ]
        do
            for arg in "${args[@]}"
            do
                python "$basedir/$module" ${arg}
            done

            # clean-up
            if [ ${#cleanup} -gt 0 ]
            then
                rm ${cleanup}
            fi
            repeat=$(($repeat - 1))
        done
    elif [ "$#" -gt 0 ]
    then
        case "$1" in
            gan)
                shift
                python "$basedir/model/gan_vgg.py" $@
                ;;
            *)
                print_help
                ;;
        esac
    else
        print_help
    fi
}

main $@
