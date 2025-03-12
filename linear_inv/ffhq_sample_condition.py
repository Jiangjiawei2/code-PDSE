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
from util.img_utils import clear_color, mask_generator, normalize_np, Blurkernel, generate_tilt_map
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
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多信息')

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
    try:
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt", **cond_config['params'])
    except FileNotFoundError:
        logger.warning("无法找到checkpoint文件，将尝试继续执行...")
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    
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
    # 使用更简洁的路径
    out_path = os.path.join(args.save_dir, 
                           measure_config['operator']['name'], 
                           task_config['data']['name'], 
                           args.algo, 
                           args.noise_type, 
                           f'noise_scale{args.noise_scale}', 
                           task_config['conditioning']['method'])
    
    # 创建所需目录
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    try:
        dataset = get_dataset(**data_config, transforms=transform)
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        raise

    # Exception) In case of inpainting, we need to generate a mask 
    mask = None
    if measure_config['operator']['name'] == 'inpainting':
        try:
            mask_gen = mask_generator(**measure_config['mask_opt'])
        except KeyError:
            logger.warning("未找到mask_opt配置，使用默认配置")
            mask_gen = mask_generator(mask_type='box', 
                                      mask_len_range=(32, 128),
                                      mask_prob_range=(0.3, 0.7),
                                      image_size=model_config['image_size'])
    
    # 设置CSV文件路径并打开文件
    out_csv_path = os.path.join(out_path, 'metrics_results.csv')
    with open(out_csv_path, mode='w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(['filename', 'psnr', 'ssim', 'lpips'])  # 写入标题行

    # 存储各样本的指标
    psnrs_list = []
    ssims_list = []
    lpipss_list = []
    execution_times = []
    
# 初始化TensorBoard SummaryWriter
    try:
        tb_log_dir = os.path.join(out_path, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to {tb_log_dir}")
        
        # 记录实验配置到TensorBoard
        writer.add_text('Experiment/Algorithm', args.algo)
        writer.add_text('Experiment/Noise_Type', args.noise_type)
        writer.add_text('Experiment/Noise_Scale', str(args.noise_scale))
        writer.add_text('Experiment/Iterations', str(args.iter))
        writer.add_text('Experiment/Timesteps', str(args.timestep))
        
        # 记录超参数到TensorBoard
        hyperparams = {
            'algorithm': args.algo,
            'noise_type': args.noise_type,
            'noise_scale': args.noise_scale,
            'iterations': args.iter,
            'timesteps': args.timestep,
            'eta': args.eta,
            'scale': args.scale,
            'iter_step': args.iter_step,
            'method': args.method
        }
        # 使用一个空字典作为指标，因为我们还没有实际的结果
        writer.add_hparams(hyperparams, {})
        
        # 添加模型和数据配置
        writer.add_text('Config/Model', str(model_config))
        writer.add_text('Config/Diffusion', str(diffusion_config))
        writer.add_text('Config/Task', str(task_config))
    except Exception as e:
        logger.error(f"初始化TensorBoard时出错: {e}")
        writer = None  # 如果出错，将writer设为None
    
    # 初始化LPIPS度量标准
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        logger.error(f"初始化LPIPS时出错: {e}")
        # 创建一个假的loss_fn_alex，避免后续代码崩溃
        class DummyLPIPS:
            def __call__(self, x, y):
                return torch.tensor([0.0]).to(device)
        loss_fn_alex = DummyLPIPS()
    
    #### 执行推理
    max_images = 10  # 限制处理的图像数量
    for i, ref_img in enumerate(loader):
        if i >= max_images:
            break
        
        logger.info(f"Inference for image {i}")
        fname = f'{i:03}.png'
        
        try:
            # 确保ref_img在正确的设备上
            ref_img = ref_img.to(device)
            
            # 记录原始参考图像到TensorBoard
            if writer is not None:
                # 规范化图像以便显示
                normalized_ref = (ref_img[0] + 1) / 2  # 从[-1,1]转换到[0,1]
                writer.add_image(f'Original/Image_{i}', normalized_ref, i)
            
            # 特殊情况处理：inpainting
            if measure_config['operator']['name'] == 'inpainting':
                try:
                    # 生成掩码
                    mask = mask_gen(ref_img)
                    mask = mask[:, 0, :, :].unsqueeze(dim=0)
                    
                    # 更新条件函数
                    measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                    sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
                    
                    # 应用测量模型 (Ax + n)
                    y = operator.forward(ref_img, mask=mask)
                    y_n = noiser(y.to(device))
                    
                    # 记录掩码到TensorBoard
                    if writer is not None:
                        writer.add_image(f'Masks/Image_{i}', mask[0], i)
                except Exception as e:
                    logger.error(f"处理inpainting时出错: {e}")
                    continue
                
            # 特殊情况处理：turbulence
            elif measure_config['operator']['name'] == 'turbulence':
                try:
                    mask = None
                    img_size = ref_img.shape[-1]
                    tilt = generate_tilt_map(img_h=img_size, img_w=img_size, kernel_size=7, device=device)
                    tilt = torch.clip(tilt, -2.5, 2.5)
                    kernel_size = task_config.get("kernel_size", 31)  # 使用默认值，防止KeyError
                    intensity = task_config.get("intensity", 3.0)
                    
                    # 模糊核
                    conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
                    kernel = conv.get_kernel().type(torch.float32)
                    kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
                    y = operator.forward(ref_img, kernel, tilt)
                    
                    y_n = noiser(y).to(device)
                    
                    # 记录扰动地图和核到TensorBoard
                    if writer is not None:
                        if tilt[0][0].dim() == 2:  # 确保维度正确
                            writer.add_image(f'Turbulence/Tilt_Map_{i}', tilt[0][0].unsqueeze(0), i)
                        if kernel[0][0].dim() == 2:  # 确保维度正确
                            writer.add_image(f'Turbulence/Kernel_{i}', kernel[0][0].unsqueeze(0), i)
                except Exception as e:
                    logger.error(f"处理turbulence时出错: {e}")
                    continue
                
            # 默认情况
            else:
                mask = None
                y = operator.forward(ref_img)
                y_n = noiser(y).to(device)
            
            # 为每个图像设置一个固定的随机种子
            random_seed = 42 + i  # 为每个图像使用不同的种子
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # 记录测量结果到TensorBoard
            if writer is not None:
                normalized_y_n = (y_n[0] + 1) / 2  # 从[-1,1]转换到[0,1]
                writer.add_image(f'Measurements/Image_{i}', normalized_y_n, i)
            
            # 根据选择的算法执行处理
            start_time = time.time()
            
            if args.algo == 'dmplug':
                try:
                    sample, metrics = DMPlug(
                        model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                        measure_config, fname, early_stopping_threshold=1e-3, stop_patience=5, out_path=out_path,
                        iteration=args.iter, lr=0.02, denoiser_step=args.timestep, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"DMPlug执行错误: {e}")
                    continue
                
            elif args.algo == 'dmplug_turbulence':
                try:
                    sample, metrics = DMPlug_turbulence(
                        model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                        measure_config, task_config, fname, kernel_ref=kernel, early_stopping_threshold=1e-3, 
                        stop_patience=5, out_path=out_path, iteration=args.iter, lr=0.02, denoiser_step=args.timestep, 
                        mask=mask, random_seed=random_seed, writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"DMPlug_turbulence执行错误: {e}")
                    continue
                
            elif args.algo == 'acce_ours':
                try:
                    sample, metrics = acce_DMPlug(
                        model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                        iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                        early_stopping_threshold=0.01, lr=0.02, out_path=out_path, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"acce_DMPlug执行错误: {e}")
                    continue
                
            elif args.algo == 'mpgd':
                try:
                    sample, metrics = mpgd(
                        sample_fn, ref_img, y_n, out_path, fname, device, 
                        mask=mask, random_seed=random_seed, writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"mpgd执行错误: {e}")
                    continue
                
            elif args.algo == 'RED_diff':
                try:
                    sample, metrics = RED_diff(
                        model, sampler, measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                        measure_config, fname, early_stopping_threshold=1e-3, stop_patience=5, out_path=out_path,
                        iteration=args.iter, lr=0.02, denoiser_step=args.timestep, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"RED_diff执行错误: {e}")
                    continue
                
            elif args.algo == 'acce_RED_diff':
                try:
                    sample, metrics = acce_RED_diff(
                        model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                        iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                        early_stopping_threshold=0.02, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"acce_RED_diff执行错误: {e}")
                    continue
                
            elif args.algo == 'acce_RED_diff_turbulence':
                try:
                    sample, metrics = acce_RED_diff_turbulence(
                        model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, task_config, operator, fname,
                        kernel_ref=kernel, iter_step=3, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                        early_stopping_threshold=0.02, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"acce_RED_diff_turbulence执行错误: {e}")
                    continue
                
            elif args.algo == 'dps':
                try:
                    sample, metrics = DPS(
                        sample_fn, ref_img, y_n, out_path, fname, device, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"dps执行错误: {e}")
                    continue
                
            elif args.algo == 'acce_RED_earlystop':
                try:
                    sample, metrics = acce_RED_earlystop(
                        model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                        iter_step=3, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                        early_stopping_threshold=0.01, lr=0.01, out_path=out_path, mask=mask, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"acce_RED_earlystop执行错误: {e}")
                    continue
                
            else:
                logger.error(f"未知算法: {args.algo}")
                continue
                
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            logger.info(f"{args.algo}执行用时: {execution_time:.2f}秒")
            
            # 记录重建结果到TensorBoard
            if writer is not None:
                # 确保样本的形状正确
                if sample.dim() == 4:
                    sample_img = sample[0]
                else:
                    sample_img = sample
                
                normalized_sample = (sample_img + 1) / 2  # 从[-1,1]转换到[0,1]
                writer.add_image(f'Reconstructions/Image_{i}', normalized_sample, i)
                
                # 记录误差图（原始图像与重建图像之间的差异）
                error_map = torch.abs(ref_img - sample)
                if error_map.dim() == 4:
                    error_map = error_map[0]
                # 规范化误差图以便显示
                error_map = error_map / (error_map.max() + 1e-8)
                writer.add_image(f'ErrorMaps/Image_{i}', error_map, i)
            
            # 记录返回的指标
            if metrics:
                psnrs_list.append(metrics.get('psnr', 0))
                ssims_list.append(metrics.get('ssim', 0))
                lpipss_list.append(metrics.get('lpips', 0))
                
                # 记录到CSV
                with open(out_csv_path, mode='a', newline='') as csv_file:
                    csvwriter = csv.writer(csv_file)
                    csvwriter.writerow([fname, metrics.get('psnr', 0), metrics.get('ssim', 0), metrics.get('lpips', 0)])
                
                # 在TensorBoard中记录每张图像的指标
                if writer is not None:
                    writer.add_scalar('Metrics/PSNR_per_image', metrics.get('psnr', 0), i)
                    writer.add_scalar('Metrics/SSIM_per_image', metrics.get('ssim', 0), i)
                    writer.add_scalar('Metrics/LPIPS_per_image', metrics.get('lpips', 0), i)
            
        except Exception as e:
            logger.error(f"处理图像{i}时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 在所有图像处理完成后，记录平均指标
    if psnrs_list:
        avg_psnr = np.mean(psnrs_list)
        avg_ssim = np.mean(ssims_list)
        avg_lpips = np.mean(lpipss_list)
        std_psnr = np.std(psnrs_list)
        std_ssim = np.std(ssims_list)
        std_lpips = np.std(lpipss_list)
        
        if writer is not None:
            writer.add_scalar('Metrics/Avg_PSNR', avg_psnr, 0)
            writer.add_scalar('Metrics/Avg_SSIM', avg_ssim, 0)
            writer.add_scalar('Metrics/Avg_LPIPS', avg_lpips, 0)
            writer.add_scalar('Metrics/Std_PSNR', std_psnr, 0)
            writer.add_scalar('Metrics/Std_SSIM', std_ssim, 0)
            writer.add_scalar('Metrics/Std_LPIPS', std_lpips, 0)
        
        # 通过箱形图展示指标分布
        try:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].boxplot(psnrs_list)
            ax[0].set_title('PSNR Distribution')
            ax[1].boxplot(ssims_list)
            ax[1].set_title('SSIM Distribution')
            ax[2].boxplot(lpipss_list)
            ax[2].set_title('LPIPS Distribution')
            plt.tight_layout()
            
            # 保存为本地文件并添加到TensorBoard
            dist_path = os.path.join(out_path, 'metric_distributions.png')
            plt.savefig(dist_path)
            
            if writer is not None and os.path.exists(dist_path):
                img = torchvision.transforms.ToTensor()(plt.imread(dist_path))
                writer.add_image('Distributions/Boxplots', img, 0)
            plt.close()
            
            # 记录平均执行时间
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            if writer is not None:
                writer.add_scalar('Performance/Avg_Execution_Time', avg_execution_time, 0)
            
            # 将图像的PSNR, SSIM, LPIPS值绘制为条形图
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            x = list(range(len(psnrs_list)))
            ax[0].bar(x, psnrs_list)
            ax[0].set_title('PSNR per Image')
            ax[0].set_xlabel('Image Index')
            ax[0].set_ylabel('PSNR (dB)')
            
            ax[1].bar(x, ssims_list)
            ax[1].set_title('SSIM per Image')
            ax[1].set_xlabel('Image Index')
            ax[1].set_ylabel('SSIM')
            
            ax[2].bar(x, lpipss_list)
            ax[2].set_title('LPIPS per Image')
            ax[2].set_xlabel('Image Index')
            ax[2].set_ylabel('LPIPS')
            
            plt.tight_layout()
            metrics_path = os.path.join(out_path, 'metrics_per_image.png')
            plt.savefig(metrics_path)
            
            if writer is not None and os.path.exists(metrics_path):
                img = torchvision.transforms.ToTensor()(plt.imread(metrics_path))
                writer.add_image('Distributions/Metrics_per_Image', img, 0)
            plt.close()
            
        except Exception as e:
            logger.error(f"生成指标可视化时出错: {e}")
    
    # 关闭TensorBoard writerfv
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer已关闭")
    
    # 计算并输出总的运行时间
    total_execution_time = sum(execution_times)
    avg_execution_time = np.mean(execution_times) if execution_times else 0
    logger.info(f"总执行时间({len(execution_times)}个图像): {total_execution_time:.2f}秒")
    logger.info(f"平均每图像执行时间: {avg_execution_time:.2f}秒")

    # 打印最终的平均指标
    if psnrs_list:
        logger.info(f"平均PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        logger.info(f"平均SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        logger.info(f"平均LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")
    
    # 输出TensorBoard查看指令
    logger.info(f"\n查看TensorBoard日志，请运行:")
    logger.info(f"tensorboard --logdir={os.path.join(out_path, 'tensorboard_logs')}")

if __name__ == '__main__':
    main()