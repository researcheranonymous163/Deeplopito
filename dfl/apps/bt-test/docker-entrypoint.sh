#!/usr/bin/bash

# For `--relative-fastresume`, see https://github.com/qbittorrent/qBittorrent/wiki/How-to-use-portable-mode.
qbittorrent-nox --relative-fastresume >/dev/null 2>&1 &

cleanup() {
	kill -SIGTERM %1
	wait %1
}
trap cleanup EXIT

sleep 1 # Wait for qBittorrent's startup.

python3.12 -u ./main.py