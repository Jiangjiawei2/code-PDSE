#!/bin/bash

# 直接执行每个包含不同参数的命令


echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = super_resolution"
echo "----------------------------------------"


echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = gaussian_deblur"
echo "----------------------------------------"

echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = motion_deblur"
echo "----------------------------------------"

echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/nonlinear_deblur_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = nonlinear_deblur"
echo "----------------------------------------"

echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = inpainting"
echo "----------------------------------------"

echo "Running experiment with algo=reddiff"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/phase_retrieval_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="reddiff"\
    --iter=300\
    --noise_type="gaussian"\
    --noise_scale=0.01
echo "Finished experiment with algo=reddiff, task = phase_retrieval"
echo "----------------------------------------"

# echo "Running experiment with algo=dmplug_turbulence"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/turbulence_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug_turbulence"\
#     --iter=1000
#     --noise_type="gaussian"\
#     --noise_scale=0.01
# echo "Finished experiment with algo=reddiff, task = bid_turbulence"
# echo "----------------------------------------"


### 3.9 unknow noise experiment 

## gaussian
# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="gaussian"\
#     --noise_scale=0.08
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"

# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="gaussian"\
#     --noise_scale=0.12
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"


# # impulse
# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="impulse"\
#     --noise_scale=0.03
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"

# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="impulse"\
#     --noise_scale=0.06
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"

# # shot
# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="shot"\
#     --noise_scale=60
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"


# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="shot"\
#     --noise_scale=25
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"


# speckle
# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="speckle"\
#     --noise_scale=0.15
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"

# ## speckle
# echo "Running experiment with algo=reddiff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="reddiff"\
#     --iter=500\
#     --noise_type="speckle"\
#     --noise_scale=0.20
# echo "Finished experiment with algo=reddiff, task = super_resolution"
# echo "----------------------------------------"