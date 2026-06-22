#!/usr/bin/env bash
# ===================== HUSE evaluation / inference script =====================
# Generates virtual IHC images for every marker and writes a PSNR/SSIM/KID report.

set -e

CHECKPOINT=PATH/TO/checkpoint-best.pth          # trained model
DATA_PATH=PATH/TO/Orion-CRC/test                # test set root: <root>/he, <root>/<marker>
OUTPUT_DIR=./orion_eval_outputs                 # generated images saved here (per marker)
OUTPUT_TXT=./orion_eval_outputs/metrics.txt     # metric report

CUDA_VISIBLE_DEVICES=0 python test_orion.py \
    --model JiT-B/16 \
    --img_size 256 \
    --num_sampling_steps 50 \
    --noise_scale 2.0 \
    --batch_size 8 \
    --checkpoint ${CHECKPOINT} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_txt ${OUTPUT_TXT}
