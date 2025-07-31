#!/usr/bin/env sh

set -e

cd "$(dirname "$0")"

if [ -z "$RUN_DIR" ]; then
	export RUN_DIR="./runs/$(date '+%Y-%m-%d %H-%M-%S-%N %Z')/"
fi

echo "RUN_DIR=$RUN_DIR"
docker compose "$@"