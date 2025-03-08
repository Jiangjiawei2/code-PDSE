#!/bin/bash

# 直接执行每个包含不同参数的命令

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --timestep=6 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="acce_RED_diff"\
    --iter=500\
    --sigma=0.01
echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.03
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"


# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.05
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.1
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --sigma=0.15
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"

