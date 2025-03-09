#!/bin/bash

# 直接执行每个包含不同参数的命令


# echo "Running experiment with algo=dmplug"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=3 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="dmplug"\
#     --iter=2000\
#     --noise_type=""
#     --sigma=0.03
# echo "Finished experiment with algo=dmplug, task = inpainting"
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
#     --iter=2000\
#     --sigma=0.05
# echo "Finished experiment with algo=dmplug, task = inpainting"
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
#     --iter=2000\
#     --sigma=0.1
# echo "Finished experiment with algo=dmplug, task = inpainting"
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
#     --iter=2000\
#     --sigma=0.15
# echo "Finished experiment with algo=dmplug, task = inpainting"
# echo "----------------------------------------"


### 3.9 unknow noise experiment 

# gaussian
echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="gaussian"\
    --noise_scale=0.08
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"

echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="gaussian"\
    --noise_scale=0.12
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"


# impulse
echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="impulse"\
    --noise_scale=0.03
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"

echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="impulse"\
    --noise_scale=0.06
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"

# shot
echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="shot"\
    --noise_scale=60
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"


echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="shot"\
    --noise_scale=25
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"


# speckle
echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="speckle"\
    --noise_scale=0.15
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"

## speckle
echo "Running experiment with algo=dmplug"
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=3 \
    --scale=17.5 \
    --method="mpgd_wo_proj" \
    --algo="dmplug"\
    --iter=2000\
    --noise_type="speckle"\
    --noise_scale=0.20
echo "Finished experiment with algo=dmplug, task = super_resolution"
echo "----------------------------------------"