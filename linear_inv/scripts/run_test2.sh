#!/bin/bash

# 直接执行每个包含不同参数的命令

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = super_resolution"
# echo "----------------------------------------"


# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/gaussian_deblur_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = gaussian_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/motion_deblur_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = motion_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/nonlinear_deblur_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = nonlinear_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=mpgd"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/phase_retrieval_config.yaml \
#     --timestep=50 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="mpgd"
# echo "Finished experiment with algo=mpgd, task = phase_retrieval"
# echo "----------------------------------------"

