#!/bin/bash

# 直接执行每个包含不同参数的命令

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=10 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.01\
#     --iter_step=1
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=10 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.01\
#     --iter_step=3
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=10 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.01\
#     --iter_step=4
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=10 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.01\
#     --iter_step=5
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=10 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="acce_RED_diff"\
    --iter=500\
    --sigma=0.01\
    --iter_step=7
echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
echo "----------------------------------------"



