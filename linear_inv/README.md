# MPGD sovling linear inverse problems

## Prerequisites
Make sure you have followed the installation instructions in the main README and have downloaded the pretrained models in the correct directories.
TODO: reiterate how to do these and write more detailed instructions

## Running Inference 
All running scripts are in `scripts/`
TODO: write some argument explanations and instructions


## super-resolution
python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/super_resolution_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_ours'



python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/super_resolution_config.yaml --timestep=4 --scale=17.5 --method="mpgd_ae" --algo='acce_ours'



python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/super_resolution_config.yaml --timestep=3 --scale=17.5 --method="ddnm" --algo='acce_ours'


python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/super_resolution_config.yaml --timestep=4 --scale=17.5 --method="vanilla" --algo='acce_ours'

## gaussian deblur

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/gaussian_deblur_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_ours'

## inpainting

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/inpainting_config.yaml --timestep=4 --scale=1 --method="mpgd_wo_proj" --algo='acce_ours'

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/inpainting_config.yaml --timestep=50 --scale=17.5 --method="mpgd_wo_proj" --algo='mpgd'



## motion deblur

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/motion_deblur_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_ours'

## nonlinear deblur

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/nonlinear_deblur_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_ours'



python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/nonlinear_deblur_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='RED_diff'

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/nonlinear_deblur_config.yaml --timestep=6 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_RED_diff'

python ffhq_sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/mgpd_diffusion_config.yaml --task_config=configs/nonlinear_deblur_config.yaml --timestep=4 --scale=17.5 --method="mpgd_wo_proj" --algo='acce_RED_diff_pro'





