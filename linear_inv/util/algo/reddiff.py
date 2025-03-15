# util/comparison_algos.py
import torch
import lpips
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard

def reddiff(
    model,
    sampler,
    measurement_cond_fn,
    ref_img,
    y_n,
    args,
    operator,
    device,
    model_config,
    measure_config,
    fname,
    early_stopping_threshold=0.01,
    stop_patience=5,
    out_path="outputs",
    iteration=2000,
    lr=0.02,
    denoiser_step=3,
    mask=None,
    random_seed=None,
    writer=None,
    img_index=None
):
    """
    RED_diff_reddiff: 实现基于REDDIFF的逆问题求解算法
    
    参数:
    - model: 扩散模型
    - sampler: 采样器
    - measurement_cond_fn: 测量条件函数
    - ref_img: 参考图像
    - y_n: 带噪声的测量
    - args: 参数集合
    - operator: 前向操作符
    - device: 运算设备
    - model_config: 模型配置
    - measure_config: 测量配置
    - fname: 输出文件名
    - early_stopping_threshold: 早停阈值
    - stop_patience: 早停耐心参数
    - out_path: 输出路径
    - iteration: 最大迭代次数
    - lr: 学习率
    - denoiser_step: 去噪器步数
    - mask: 可选掩码
    - random_seed: 随机种子
    - writer: TensorBoard写入器
    - img_index: 图像索引
    
    返回:
    - 重建图像
    - 评价指标
    """
    # 设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 初始化变量
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    
    # 初始化评价指标计算工具
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    
    # REDDIFF特有的自适应权重参数
    grad_term_weight = 0.25    # 对应REDDIFF中的grad_term_weight
    obs_weight = 1.0           # 对应REDDIFF中的obs_weight
    denoise_term_weight = "sqrt"  # 权重类型: "linear", "sqrt", "square", "log", "trunc_linear", "power2over3", "const"
    
    # 初始化优化器 (类似于REDDIFF中的设置)
    optimizer = torch.optim.Adam([Z], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    
    # 用于记录指标和优化过程
    losses = []
    psnrs = []
    ssims = []
    lpipss = []
    best_psnr = 0
    best_img = None
    best_epoch = 0
    
    # 主优化循环
    for epoch in tqdm(range(iteration), desc="RED_diff_reddiff Optimization"):
        model.eval()
        optimizer.zero_grad()
        
        # 使用扩散模型的去噪过程获取噪声预测和x0预测
        x_t = Z
        
        for i, t in enumerate(list(range(denoiser_step))[::-1]):
            time = torch.tensor([t] * ref_img.shape[0], device=device)
            
            # 使用现有的p_sample获取预测结果
            out = sampler.p_sample(model=model, x=x_t, t=time)
            sample = out['sample']
            pred_x0 = out['pred_xstart']
            
            # 计算噪声预测
            eps = sampler.predict_eps_from_x_start(x_t, time, pred_x0)
            
            # 计算SNR并应用权重函数
            alpha_t = sampler.alphas_cumprod[time[0]]
            snr_inv = (1-alpha_t).sqrt()/alpha_t.sqrt()
            
            if denoise_term_weight == "linear":
                w_t = grad_term_weight * snr_inv
            elif denoise_term_weight == "sqrt":
                w_t = grad_term_weight * torch.sqrt(snr_inv)
            elif denoise_term_weight == "square":
                w_t = grad_term_weight * torch.square(snr_inv)
            elif denoise_term_weight == "log":
                w_t = grad_term_weight * torch.log(snr_inv + 1.0)
            elif denoise_term_weight == "trunc_linear":
                w_t = grad_term_weight * torch.clip(snr_inv, max=1.0)
            elif denoise_term_weight == "power2over3":
                w_t = grad_term_weight * torch.pow(snr_inv, 2/3)
            elif denoise_term_weight == "const":
                w_t = grad_term_weight * torch.ones_like(snr_inv)
            
            # 计算观测误差
            if mask is not None:
                e_obs = y_n - operator.forward(Z, mask=mask)
            else:
                e_obs = y_n - operator.forward(Z)
            
            # 计算损失函数 (REDDIFF风格)
            loss_obs = (e_obs**2).mean()/2
            
            # 使用预测的噪声与Z的点积作为先验损失
            loss_noise = torch.mul(eps.detach(), Z).mean()
            
            # 总损失
            loss = w_t * loss_noise + obs_weight * loss_obs
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # 计算当前重建的评价指标
        with torch.no_grad():
            current_img_np = Z.cpu().squeeze().detach().numpy()
            if current_img_np.shape[0] == 3:  # [C, H, W]
                current_img_np = np.transpose(current_img_np, (1, 2, 0))
                
            ref_img_np = ref_img.cpu().squeeze().numpy()
            if ref_img_np.shape[0] == 3:  # [C, H, W]
                ref_img_np = np.transpose(ref_img_np, (1, 2, 0))
            
            # 计算PSNR
            current_psnr = peak_signal_noise_ratio(ref_img_np, current_img_np)
            psnrs.append(current_psnr)
            
            # 计算SSIM
            current_ssim = structural_similarity(ref_img_np, current_img_np, channel_axis=2, data_range=1)
            ssims.append(current_ssim)
            
            # 计算LPIPS
            current_img_torch = torch.from_numpy(current_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            current_lpips = loss_fn_alex(ref_img_torch, current_img_torch).item()
            lpipss.append(current_lpips)
            
            # 记录最佳PSNR图像
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_img = Z.clone()
                best_epoch = epoch
                
            # TensorBoard记录
            if writer is not None and img_index is not None:
                writer.add_scalar(f'RED_diff_reddiff/Image_{img_index}/Loss', loss.item(), epoch)
                writer.add_scalar(f'RED_diff_reddiff/Image_{img_index}/PSNR', current_psnr, epoch)
                writer.add_scalar(f'RED_diff_reddiff/Image_{img_index}/SSIM', current_ssim, epoch)
                writer.add_scalar(f'RED_diff_reddiff/Image_{img_index}/LPIPS', current_lpips, epoch)
                
                if epoch % 100 == 0 or epoch == iteration - 1:
                    writer.add_image(f'RED_diff_reddiff/Image_{img_index}/Progress', 
                                    (Z[0].clamp(-1, 1) + 1)/2, epoch)
        
        # 早停检查
        if epoch > stop_patience:
            if len(psnrs) > stop_patience:
                recent_psnrs = psnrs[-stop_patience:]
                if max(recent_psnrs) - recent_psnrs[-1] > early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch+1}: PSNR decreasing")
                    break
    
    # 如果没有找到最佳图像，使用最后一个
    if best_img is None:
        best_img = Z
    
    # 保存最终图像
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(best_img))
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    # 计算最终指标
    best_img_np = best_img.cpu().squeeze().detach().numpy()
    if best_img_np.shape[0] == 3:  # [C, H, W]
        best_img_np = np.transpose(best_img_np, (1, 2, 0))
        
    ref_img_np = ref_img.cpu().squeeze().numpy()
    if ref_img_np.shape[0] == 3:  # [C, H, W]
        ref_img_np = np.transpose(ref_img_np, (1, 2, 0))
    
    final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
    final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
    
    best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()
    
    # 记录最终指标
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    print(f"Final metrics between reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return best_img, final_metric