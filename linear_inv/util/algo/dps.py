# util/algo/mpgd.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips
import time
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard


def DPS(
    sample_fn, 
    ref_img, 
    y_n, 
    out_path, 
    fname, 
    device, 
    mask=None, 
    random_seed=None, 
    writer=None, 
    img_index=None
):
    """
    DPS (Diffusion Posterior Sampling) 算法实现
    
    采样、计算评价指标并保存结果
    """
    
    # 使用传入的随机种子重新设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 开始采样 - 传递writer和img_index参数
    x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
    sample = sample_fn(
        x_start=x_start, 
        measurement=y_n, 
        record=True, 
        save_root=out_path, 
        mask=mask, 
        ref_img=ref_img,
        writer=writer, 
        img_index=img_index
    )

    # 初始化评价指标
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # 保存结果图像
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample)) 
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    # 转换最佳图像和参考图像为 numpy 格式
    best_img_np = sample.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
    ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

    # 计算 PSNR
    final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
    # 计算 SSIM
    final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
    # 计算 LPIPS
    best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

    # 记录最终指标
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 将最终指标记录到TensorBoard
    if writer is not None and img_index is not None:
        writer.add_scalar(f'DPS/Final/PSNR', final_psnr, img_index)
        writer.add_scalar(f'DPS/Final/SSIM', final_ssim, img_index)
        writer.add_scalar(f'DPS/Final/LPIPS', final_lpips, img_index)
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, final_metric