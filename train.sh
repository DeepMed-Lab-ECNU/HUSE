#!/usr/bin/env bash
# ===================== HUSE training launch script =====================
# Everything is configured in the YAML, including how many GPUs to use (`gpus`).
# Just point this script at a config:
#   bash train.sh configs/orion.yaml

set -e

CONFIG=${1:-configs/orion.yaml}

# Read gpus / master_port / output_dir from the YAML config.
GPUS=$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}')).get('gpus','0'))")
MASTER_PORT=$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}')).get('master_port',29474))")
OUTPUT_DIR=$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}')).get('output_dir','./output'))")
NPROC=$(echo ${GPUS} | awk -F',' '{print NF}')

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=${MASTER_PORT} \
    main.py \
    --config ${CONFIG} \
    > ${OUTPUT_DIR}/train.out 2>&1
