#!/usr/bin/env bash

python tools/train.py \
    local_configs/topformer/topformer_small_512x512_20k_2x8_imaterialistfashion3.py \
    --gpus 1 \
    --work-dir runs \
    --seed 42 \

