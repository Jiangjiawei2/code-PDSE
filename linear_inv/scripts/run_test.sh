#!/bin/bash

# 直接执行每个包含不同参数的命令

# echo "Running experiment with algo=acce_RED_diff"   ### lr :0.01,  p = 1.5  # 控制衰减的速度，p 越大，衰减越快   幂次衰减
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/gaussian_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500
# echo "Finished experiment with algo=acce_RED_diff, task = gaussian_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/motion_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500
# echo "Finished experiment with algo=acce_RED_diff, task = motion_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/nonlinear_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500
# echo "Finished experiment with algo=acce_RED_diff, task = nonlinear_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"

echo "Running experiment with algo=acce_RED_diff_turbulence"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/turbulence_config.yaml \
    --timestep=6 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="acce_RED_diff_turbulence"\
    --iter=800
echo "Finished experiment with algo=acce_RED_diff_turbulence, task = bid_turbulence"
echo "----------------------------------------"
