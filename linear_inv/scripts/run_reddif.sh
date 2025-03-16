#!/bin/bash

# 直接执行每个包含不同参数的命令

echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=1000 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=1000
echo "Finished experiment with algo=reddiff, task = super_resolution"
echo "----------------------------------------"

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --timestep=6 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=500
echo "Finished experiment with algo=reddiff, task = gaussian_deblur"
echo "----------------------------------------"

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --timestep=6 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=500
echo "Finished experiment with algo=reddiff, task = motion_deblur"
echo "----------------------------------------"

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/nonlinear_deblur_config.yaml \
    --timestep=6 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=500
echo "Finished experiment with algo=reddiff, task = nonlinear_deblur"
echo "----------------------------------------"

python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --timestep=6 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=500
echo "Finished experiment with algo=reddiff, task = inpainting"
echo "----------------------------------------"

# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/turbulence_config.yaml \
#     --timestep=6 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="RED_diff_reddiff_turbulence"\
#     --iter=800
# echo "Finished experiment with algo=RED_diff_reddiff_turbulence, task = bid_turbulence"
# echo "----------------------------------------"
