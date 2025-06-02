#!/usr/bin/env bash

set -xe

if [ $# -eq 0 ]; then
    echo "Usage: $0 user@hostname [port] [remote_directory]"
    echo "Example: $0 john@server.com"
    echo "Example: $0 john@server.com 8080 ~/my-project"
    exit 1
fi

USER_HOST="$1"
SSH_PORT="$2"
PORT="${3:-2718}"

ssh -N -i ~/.ssh/runpod \
    -p "$SSH_PORT" \
    -L "0.0.0.0:$PORT:127.0.0.1:$PORT" \
    -o "ExitOnForwardFailure=yes" \
    "$USER_HOST" -vvv
