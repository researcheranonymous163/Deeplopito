#!/usr/bin/env sh

set -e

if [ -z "${NO_TC+x}" ]; then
	# Limit the egress traffic rate to 256 KiB/s.
	tc qdisc add dev eth0 root handle 1: tbf rate ${UPLOAD_BANDWIDTH} burst 16k latency 50ms

	# Limit the ingress traffic rate to 2 MiB/s.
	tc qdisc add dev eth0 ingress
	tc filter add dev eth0 ingress protocol all u32 match u32 0 0 police rate ${DOWNLOAD_BANDWIDTH} burst 256k drop
fi


if [ "${CAPTURE_EXECUTION_TIME}" = "True" ];then
  python -m simulateModel.runTimeRecorderBert
fi

if [ "${RUN_MAIN_PROGRAM}" = "True" ];then
  python -m simulateModel.algorithm
fi