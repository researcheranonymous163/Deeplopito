#!/usr/bin/env sh

set -e

cd "$(dirname "$0")"

./compose.sh -f ./compose-grpc.yaml "$@"