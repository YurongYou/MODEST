#!/usr/bin/env bash

set -e
set -x

NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

while true
do
    END_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $END_PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

TORCHELASTIC_MAX_RESTARTS=0 python -m torch.distributed.launch --rdzv_backend=c10d --rdzv_endpoint=localhost:$END_PORT --nproc_per_node=${NGPUS} train.py --launcher pytorch --world_size=${NGPUS} --tcp_port ${PORT} ${PY_ARGS}

