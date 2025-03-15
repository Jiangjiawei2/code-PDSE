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
def mpgd(
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
    MPGD算法实现：使用扩散模型进行线性逆问题求解
    
    Parameters:
    - sample_fn: 采样函数，通常是sampler.p_sample_loop的偏函数
    - ref_img: 参考图像张量 [B, C, H, W]
    - y_n: 带噪声的测量张量 [B, C, H, W]
    - out_path: 输出保存路径
    - fname: 保存的文件名
    - device: 运行的设备（CPU 或 GPU）
    - mask: 可选的掩码张量（用于inpainting问题）
    - random_seed: 随机种子，确保可重复性
    - writer: TensorBoard SummaryWriter对象
    - img_index: 当前处理的图像索引，用于TensorBoard日志
    
    Returns:
    - sample: 重建的图像张量
    - metrics: 包含PSNR、SSIM和LPIPS的指标字典
    """
    import os
    import time
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from tqdm import tqdm
    
    # 使用传入的随机种子重新设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # TensorBoard日志：记录实验配置
    if writer is not None and img_index is not None:
        writer.add_text(f'MPGD/Image_{img_index}/Config', 
                       f'Random Seed: {random_seed}\n'
                       f'Mask: {"Yes" if mask is not None else "No"}', 0)
        
        # 记录参考图像和测量图像
        writer.add_image(f'MPGD/Input/Image_{img_index}', (y_n[0].clamp(-1, 1) + 1)/2, 0)
        writer.add_image(f'MPGD/Reference/Image_{img_index}', (ref_img[0].clamp(-1, 1) + 1)/2, 0)
    
    # 开始采样
    start_time = time.time()
    
    # 使用随机噪声初始化
    x_start = torch.randn_like(ref_img, device=device)
    
    # 使用进度条包装采样过程
    pbar = tqdm(total=1, desc=f"MPGD Sampling for image {img_index if img_index is not None else 'N/A'}")
    
    # 调用采样函数
    try:
        sample = sample_fn(
            x_start=x_start, 
            measurement=y_n, 
            record=True, 
            save_root=out_path, 
            mask=mask,
            ref_img=ref_img,  # 添加参考图像
            writer=writer,    # 添加writer
            img_index=img_index  # 添加图像索引
        )
    except Exception as e:
        print(f"采样函数执行错误: {e}")
        # 如果采样函数不支持这些参数，退回到基本版本
        sample = sample_fn(
            x_start=x_start, 
            measurement=y_n, 
            record=True, 
            save_root=out_path, 
            mask=mask
        )
    
    # 更新进度条
    pbar.update(1)
    pbar.close()
    
    # 计算采样时间
    sampling_time = time.time() - start_time
    
    # 确保输出目录存在
    os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'input'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)

    # 初始化评价指标计算
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # 保存结果图像
    try:
        # 使用安全的图像保存函数
        def save_tensor_image(tensor, path):
            """保存张量图像到文件"""
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 确保是CPU张量并分离梯度
            img = tensor.detach().cpu()
            
            # 转换为numpy数组
            if img.dim() == 4:  # [B, C, H, W]
                img = img.squeeze(0)  # 转换为[C, H, W]
            
            img_np = img.numpy()
            
            # 调整通道顺序和归一化
            if img_np.shape[0] == 3:  # [C, H, W]
                img_np = np.transpose(img_np, (1, 2, 0))  # 转换为[H, W, C]
            
            # 规范化到[0, 1]范围
            img_np = (img_np + 1) / 2  # 从[-1,1]转换到[0,1]
            img_np = np.clip(img_np, 0, 1)  # 确保在[0,1]范围内
            
            # 保存图像
            plt.imsave(path, img_np)
            
        # 保存图像
        save_tensor_image(sample, os.path.join(out_path, 'recon', fname))
        save_tensor_image(y_n, os.path.join(out_path, 'input', fname))
        save_tensor_image(ref_img, os.path.join(out_path, 'label', fname))
    except Exception as e:
        print(f"保存图像失败: {e}")
        try:
            # 备选方案
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample)) 
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        except Exception as e2:
            print(f"备选保存方案也失败: {e2}")
    
    # 计算评价指标
    psnr_val = 0
    ssim_val = 0
    lpips_val = 0
    
    try:
        # 转换最佳图像和参考图像为 numpy 格式
        sample_np = sample.detach().cpu().squeeze().numpy()
        ref_np = ref_img.cpu().squeeze().numpy()
        
        # 确保格式正确 [H, W, C]
        if sample_np.shape[0] == 3:  # [C, H, W]
            sample_np = np.transpose(sample_np, (1, 2, 0))
        if ref_np.shape[0] == 3:  # [C, H, W]
            ref_np = np.transpose(ref_np, (1, 2, 0))
            
        # 计算 PSNR
        psnr_val = peak_signal_noise_ratio(ref_np, sample_np)
        
        # 计算 SSIM
        ssim_val = structural_similarity(ref_np, sample_np, channel_axis=2, data_range=1)
        
        # 计算 LPIPS
        # 准备用于LPIPS计算的张量
        sample_tensor = torch.from_numpy(sample_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # 确保值范围在[-1, 1]
        if sample_tensor.min() >= 0 and sample_tensor.max() <= 1:
            sample_tensor = sample_tensor * 2 - 1
        if ref_tensor.min() >= 0 and ref_tensor.max() <= 1:
            ref_tensor = ref_tensor * 2 - 1
            
        lpips_val = loss_fn_alex(sample_tensor, ref_tensor).item()
    except Exception as e:
        print(f"计算指标时出错: {e}")
        import traceback
        traceback.print_exc()

    # 记录最终指标
    metrics = {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    }
    
    # 将最终指标记录到TensorBoard
    if writer is not None and img_index is not None:
        # 记录性能指标
        writer.add_scalar(f'MPGD/Performance/SamplingTime', sampling_time, img_index)
        writer.add_scalar(f'MPGD/Metrics/PSNR', psnr_val, img_index)
        writer.add_scalar(f'MPGD/Metrics/SSIM', ssim_val, img_index)
        writer.add_scalar(f'MPGD/Metrics/LPIPS', lpips_val, img_index)
        
        # 记录重建图像
        writer.add_image(f'MPGD/Reconstructed/Image_{img_index}', (sample[0].clamp(-1, 1) + 1)/2, img_index)
        
        # 尝试记录误差图像
        try:
            error_map = torch.abs(ref_img - sample)
            error_map = error_map / error_map.max()  # 归一化误差
            writer.add_image(f'MPGD/ErrorMap/Image_{img_index}', error_map[0], img_index)
        except Exception as e:
            print(f"记录误差图像失败: {e}")
    
    # 打印最终的性能指标
    print(f"MPGD Performance for Image {img_index if img_index is not None else 'N/A'}:")
    print(f"Sampling Time: {sampling_time:.4f} seconds")
    print(f"Metrics - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
    
    return sample, metrics

