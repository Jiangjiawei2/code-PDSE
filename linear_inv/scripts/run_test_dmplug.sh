#!/bin/bash

# 直接执行每个包含不同参数的命令


# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000

# echo "Finished experiment with algo=dmplug, task = super_resolution"
# echo "----------------------------------------"


# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/gaussian_deblur_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000

# echo "Finished experiment with algo=dmplug, task = gaussian_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/motion_deblur_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000
# echo "Finished experiment with algo=dmplug, task = motion_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/nonlinear_deblur_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000
# echo "Finished experiment with algo=dmplug, task = nonlinear_deblur"
# echo "----------------------------------------"

# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000
# echo "Finished experiment with algo=dmplug, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/phase_retrieval_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=1000
# echo "Finished experiment with algo=dmplug, task = phase_retrieval"
# echo "----------------------------------------"

echo "Running experiment with algo=dmplug_turbulence"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/turbulence_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug_turbulence"\
    --iter=2000
echo "Finished experiment with algo=dmplug, task = bid_turbulence"
echo "----------------------------------------"



