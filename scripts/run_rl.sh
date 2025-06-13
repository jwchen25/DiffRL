#!/bin/bash

source ~/software/anaconda3/bin/activate matrl
# source ~/work/code/mattergen/.venv/bin/activate
export HYDRA_FULL_ERROR=1
# source .env

EXPNAME="matgen_bg"
# EXPNAME="reinmat_test"

nohup python -u main.py \
    expname=${EXPNAME} \
    pipeline=mat_invent \
    model=mattergen \
    reward=band_gap \
    logger=wandb \
    device=cuda:0 \
    > exp_res/${EXPNAME}.log 2>&1 &
