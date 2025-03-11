from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.guided_gaussian_diffusion import create_sampler, space_timesteps
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, normalize_np, Blurkernel ,generate_tilt_map
from util.logger import get_logger
from util.tools import early_stopping
import torchvision
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv
import numpy as np
from util.pnp_test import *
from motionblur.motionblur import Kernel
from torch.utils.tensorboard import SummaryWriter

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,default="./configs/model_config.yaml")
    parser.add_argument('--diffusion_config', type=str, default="./configs/mgpd_diffusion_config.yaml")
    parser.add_argument('--task_config', type=str, default="./configs/super_resolution_config.yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--scale', type=float, default=17.5)
    parser.add_argument('--method', type=str, default='mpgd_wo_proj') # mpgd_wo_proj
    parser.add_argument('--save_dir', type=str, default='./outputs/ffhq/')
    parser.add_argument('--algo', type=str, default='acce_RED_diff')
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--noise_scale', type=float, default=0.03, help='a value of noise_scale')
    parser.add_argument('--noise_type', type=str, default='impulse', help='unkown noise type')
    parser.add_argument('--iter_step', type=float, default=3, help='New value for iter_step')

    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = f"1000"
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    task_config["measurement"]["noise"]["noise_scale"] = args.noise_scale
    task_config["measurement"]["noise"]["name"] = args.noise_type

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt", **cond_config['params']) # in the paper we used this checkpoint
    # cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/ffhq256/model.ckpt", **cond_config['params']) # you can probably also use this checkpoint, but you probably want to tune the hyper-parameter a bit
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    if args.algo == 'dps' or args.algo == 'mcg':
        sample_fn = partial(sampler.p_sample_loop_dps, model=model, measurement_cond_fn=measurement_cond_fn)
    else:
        # 修改这里，确保sample_fn也可以接收img_index参数
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    dir_path = f"{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"
    # out_path = os.path.join(args.save_dir, measure_config['operator']['name'], task_config['data']['name'], args.algo, task_config['conditioning']['method'])    
    # out_path = os.path.join(args.save_dir, measure_config['operator']['name'], task_config['data']['name'], args.algo, str(args.noise_type), task_config['conditioning']['method'])   ## noise  
    ## abltion
    # out_path = os.path.join(args.save_dir, measure_config['operator']['name'], task_config['data']['name'], args.algo, 'step'+ str(args.iter_step), task_config['conditioning']['method'])   ## noise  
    ## unkown_noise
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], task_config['data']['name'], args.algo, str(args.noise_type), 'noise_scale'+str(args.noise_scale), task_config['conditioning']['method'])   ## noise  

    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    
    # 设置CSV文件路径并打开文件
    out_csv_path = os.path.join(out_path, 'metrics_results.csv')
    csv_file = open(out_csv_path, mode='w', newline='')
    csvwriter = csv.writer(csv_file)
    csvwriter.writerow(['filename', 'psnr', 'ssim', 'lpips'])  # 写入标题行

    # 存储各样本的指标
    psnrs_list = []
    ssims_list = []
    lpipss_list = []
    
    execution_times = []
    
    # 在外部初始化 TensorBoard SummaryWriter
    # 修改这里，添加更多有用的信息到日志目录
    tb_log_dir = os.path.join(out_path, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logging to {tb_log_dir}")
    
    # 记录实验配置到TensorBoard
    writer.add_text('Experiment/Algorithm', args.algo)
    writer.add_text('Experiment/Noise_Type', args.noise_type)
    writer.add_text('Experiment/Noise_Scale', str(args.noise_scale))
    writer.add_text('Experiment/Iterations', str(args.iter))
    writer.add_text('Experiment/Timesteps', str(args.timestep))
        
    #### Do Inference
    for i, ref_img in enumerate(loader):
        if i >= 10:
            break
        logger.info(f"Inference for image {i}")
        fname = f'{i:03}.png'
        ref_img = ref_img.to(device)
        
        # Exception) In case of inpainting
        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            # 确保sample_fn可以传递img_index
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y.to(device))
        elif measure_config['operator']['name'] == 'turbulence':
            mask = None
            img_size = ref_img.shape[-1]
            tilt = generate_tilt_map(img_h=img_size, img_w=img_size, kernel_size=7, device=device)
            tilt = torch.clip(tilt, -2.5, 2.5)
            kernel_size = task_config["kernel_size"]
            intensity = task_config["intensity"]

            # blur kernel
            conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
            kernel = conv.get_kernel().type(torch.float32)
            kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
            y = operator.forward(ref_img, kernel, tilt)
        
            y_n = noiser(y).to(device)    
        else:
            mask = None
            y = operator.forward(ref_img)
            y_n = noiser(y).to(device)
      
        # 定义一个函数来设置随机种子
        def set_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 固定随机种子，例如使用 42
        random_seed = 42
        set_seed(random_seed)
        
        # 为当前图像记录图像到TensorBoard
        writer.add_image(f'Images/Original_{i}', ref_img[0], i)
        writer.add_image(f'Images/Measurement_{i}', y_n[0], i)

        # 根据传入的算法选择执行
        if args.algo == 'dmplug':
            start_time = time.time()
            sample, metrics = DMPlug(
                model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                measure_config, fname, early_stopping_threshold=1e-3, stop_patience=5, out_path=out_path,
                iteration=args.iter, lr=0.02, denoiser_step=args.timestep, mask=mask, random_seed=random_seed, 
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"DMPlug execution for image {i} took {execution_time:.2f} seconds.")
        
        elif args.algo == 'dmplug_turbulence':
            start_time = time.time()
            sample, metrics = DMPlug_turbulence(
                model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                measure_config, task_config, fname, kernel_ref=kernel, early_stopping_threshold=1e-3, 
                stop_patience=5, out_path=out_path, iteration=args.iter, lr=0.02, denoiser_step=args.timestep, 
                mask=mask, random_seed=random_seed, writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"DMPlug_turbulence execution for image {i} took {execution_time:.2f} seconds.")
            
        elif args.algo == 'acce_ours':
            start_time = time.time()
            sample, metrics = acce_DMPlug(
                model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                iter_step=args.iter_step, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                early_stopping_threshold=0.01, lr=0.02, out_path=out_path, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"acce_DMPlug execution for image {i} took {execution_time:.2f} seconds.")
            
        elif args.algo == 'mpgd':
            start_time = time.time()
            # 传递当前图像索引i用于TensorBoard日志
            sample, metrics = mpgd(
                sample_fn, ref_img, y_n, out_path, fname, device, 
                mask=mask, random_seed=random_seed, writer=writer, img_index=i
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"mpgd execution for image {i} took {execution_time:.2f} seconds.")
            
        elif args.algo == 'RED_diff':
            start_time = time.time()
            sample, metrics = RED_diff(
                model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                measure_config, fname, early_stopping_threshold=1e-3, stop_patience=5, out_path=out_path,
                iteration=args.iter, lr=0.02, denoiser_step=args.timestep, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"RED_diff execution for image {i} took {execution_time:.2f} seconds.")

        elif args.algo == 'acce_RED_diff':
            start_time = time.time()
            sample, metrics = acce_RED_diff(
                model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                early_stopping_threshold=0.02, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"acce_RED_diff execution for image {i} took {execution_time:.2f} seconds.")

        elif args.algo == 'acce_RED_diff_turbulence':
            start_time = time.time()
            sample, metrics = acce_RED_diff_turbulence(
                model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, task_config, operator, fname,
                kernel_ref=kernel, iter_step=3, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                early_stopping_threshold=0.02, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"acce_RED_diff_turbulence execution for image {i} took {execution_time:.2f} seconds.")
            
        elif args.algo == 'dps':
            start_time = time.time()
            sample, metrics = DPS(
                sample_fn, ref_img, y_n, out_path, fname, device, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"dps execution for image {i} took {execution_time:.2f} seconds.")

        elif args.algo == 'acce_RED_earlystop':
            start_time = time.time()
            sample, metrics = acce_RED_earlystop(
                model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                iter_step=3, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                early_stopping_threshold=0.01, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                writer=writer, img_index=i  # 传递图像索引
            )
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"acce_RED_earlystop execution for image {i} took {execution_time:.2f} seconds.")
            
        # 记录返回的指标
        psnrs_list.append(metrics['psnr'])
        ssims_list.append(metrics['ssim'])
        lpipss_list.append(metrics['lpips'])
        
        # 记录到CSV
        csvwriter.writerow([fname, metrics['psnr'], metrics['ssim'], metrics['lpips']])
        
        # 在TensorBoard中记录每张图像的指标
        writer.add_scalar('Metrics/PSNR_per_image', metrics['psnr'], i)
        writer.add_scalar('Metrics/SSIM_per_image', metrics['ssim'], i)
        writer.add_scalar('Metrics/LPIPS_per_image', metrics['lpips'], i)
        
    # 在所有图像处理完成后，记录平均指标
    avg_psnr = np.mean(psnrs_list)
    avg_ssim = np.mean(ssims_list)
    avg_lpips = np.mean(lpipss_list)
    
    writer.add_scalar('Metrics/Avg_PSNR', avg_psnr, 0)
    writer.add_scalar('Metrics/Avg_SSIM', avg_ssim, 0)
    writer.add_scalar('Metrics/Avg_LPIPS', avg_lpips, 0)
    
    # 记录平均执行时间
    avg_execution_time = np.mean(execution_times) if execution_times else 0
    writer.add_scalar('Performance/Avg_Execution_Time', avg_execution_time, 0)
    
    # 在训练结束时关闭 SummaryWriter
    writer.close()
    logger.info("TensorBoard writer closed")
    
    # 计算并输出总的运行时间
    total_execution_time = sum(execution_times)  # 计算总时间
    print(f"Total execution time for {len(execution_times)} images: {total_execution_time:.2f} seconds.")
    print(f"Average execution time per image: {avg_execution_time:.2f} seconds.")

    # 关闭 CSV 文件
    csv_file.close()
    # 计算并打印平均值
    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}")
    
    # 输出TensorBoard查看指令
    print(f"\nTo view the TensorBoard logs, run:")
    print(f"tensorboard --logdir={tb_log_dir}")
    
    
    
if __name__ == '__main__':
    main()
