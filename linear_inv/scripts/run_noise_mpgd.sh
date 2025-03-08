#!/bin/bash

# 直接执行每个包含不同参数的命令



echo "Running experiment with algo=mpgd"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --timestep=50 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="mpgd"\
    --sigma=0.01
echo "Finished experiment with algo=mpgd, task = inpainting"
echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"\
#     --sigma=0.03
# echo "Finished experiment with algo=mpgd, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"\
#     --sigma=0.05
# echo "Finished experiment with algo=mpgd, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"\
#     --sigma=0.1
# echo "Finished experiment with algo=mpgd, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"\
#     --sigma=0.15
# echo "Finished experiment with algo=mpgd, task = inpainting"
# echo "----------------------------------------"




