#!/bin/bash

set -e

run_bootMe() {
    echo "---> Starting nginx"
    nginx
    echo "| Done"
    echo "---> Starting gunicorn"
    gunicorn -c config/gunicorn/dev.py
    echo "| Done"
    while :
    do 
        sleep 100
    done
}

case "$1" in
run)
    shift 1
    run_bootMe "$@"
    ;;
*)
    exac "$@"
    ;;
esac