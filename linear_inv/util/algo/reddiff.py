import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color


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
    REDDIFF算法实现
    
    基于Algorithm 1 Variational sampler (RED-diff)的实现:
    Input: y, f_θ, σ_v, L, {α_t, σ_t, λ_t}_{t=1}^T
    Initialize: μ_0
    for l = 1, ..., L do
        t ~ U[0, T]
        ε ~ N(0, I_n)
        x_t = α_t·μ + σ_t·ε
        loss = ||y - f(μ)||^2 + λ_t·(sg[ε_θ(x_t; t) - ε])^T·μ
        μ ← OptimizerStep(loss)
    end for
    Return: μ
    
    参数:
    - model: 扩散模型 f_θ
    - sampler: 采样器
    - measurement_cond_fn: 测量条件函数
    - ref_img: 参考图像
    - y_n: 带噪声的测量 y
    - args: 参数集合
    - operator: 前向操作符 f
    - device: 运算设备
    - model_config: 模型配置
    - measure_config: 测量配置
    - fname: 输出文件名
    - out_path: 输出路径
    - iteration: 最大迭代次数 L
    - lr: 学习率
    - denoiser_step: 扩散模型步数
    - mask: 可选掩码
    - random_seed: 随机种子
    - writer: TensorBoard写入器
    - img_index: 图像索引
    
    返回:
    - 重建图像 μ
    - 评价指标字典
    """
    # 设置随机种子以确保结果可复现
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 初始化评价指标计算工具
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    
    # 初始化变量：μ是要优化的潜在变量
    # mu = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    initx=operator.transpose(y_n)
    mu = initx.clone().detach().requires_grad_(True)    # 初始化优化器
    optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    
    # 算法中λ_t参数 (噪声预测项权重)
    grad_term_weight = getattr(args, 'grad_term_weight', 0.25)
    
    # 用于记录指标
    losses = []
    psnrs = []
    ssims = []
    lpipss = []
    
    # 记录初始状态到TensorBoard
    if writer is not None and img_index is not None:
        writer.add_image(f'REDDIFF/Image_{img_index}/Initial', (mu[0] + 1)/2, 0)
    
    # 主优化循环 (for l = 1, ..., L)
    pbar = tqdm(range(iteration), desc="REDDIFF Optimization")
    for l in pbar:
        model.eval()
        optimizer.zero_grad()
        
        # 随机采样时间步 t ~ U[0, T]
        t = torch.randint(0, sampler.num_timesteps, (1,), device=device)[0]
        time = torch.tensor([t] * mu.shape[0], device=device)
        
        # 获取噪声调度参数
        alpha_t = torch.tensor(sampler.alphas_cumprod[t], device=device).float()
        sigma_t = torch.sqrt(1 - alpha_t)
        
        # 采样噪声 ε ~ N(0, I_n)
        epsilon = torch.randn_like(mu)
        
        noise_x0 = torch.randn_like(mu)
        
        x0_pred = mu + 0.0001*noise_x0
        # 构造噪声数据点 x_t = α_t·μ + σ_t·ε
        x_t = alpha_t.sqrt() * x0_pred + sigma_t * epsilon
        
        # 通过扩散模型获取噪声预测
        with torch.no_grad():
            # 使用p_mean_variance获取预测
            out = sampler.p_mean_variance(model, x_t, time)
            pred_xstart = out['pred_xstart']
            
            # 从x0预测推导噪声预测
            et = sampler.predict_eps_from_x_start(x_t, time, pred_xstart)
        
        # 计算数据一致性损失 ||y - f(μ)||^2
        if mask is not None:
            data_fidelity = criterion(operator.forward(x0_pred, mask=mask), y_n)
        else:
            data_fidelity = criterion(operator.forward(x0_pred), y_n)
        
        # 计算先验损失 λ_t·(sg[ε_θ(x_t; t) - ε])^T·μ
        # sg[·]表示stop gradient，即et.detach()
        prior_loss = torch.mul((et.detach() - epsilon), pred_xstart).mean()
        
        # 信噪比相关的权重调整
        snr = alpha_t / sigma_t
        lambda_t = grad_term_weight / snr
        
        # 总损失
        loss = data_fidelity + lambda_t * prior_loss
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # 计算当前重建的评价指标
        with torch.no_grad():
            current_img_np = mu.cpu().squeeze().detach().numpy()
            if current_img_np.shape[0] == 3:  # 转换为[H, W, C]格式
                current_img_np = np.transpose(current_img_np, (1, 2, 0))
                
            ref_img_np = ref_img.cpu().squeeze().numpy()
            if ref_img_np.shape[0] == 3:
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
            
            # 更新进度条信息
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{current_psnr:.2f}", 
                'SSIM': f"{current_ssim:.4f}"
            })
            
            # TensorBoard记录
            if writer is not None and img_index is not None and l % 50 == 0:
                writer.add_scalar(f'REDDIFF/Image_{img_index}/Loss', loss.item(), l)
                writer.add_scalar(f'REDDIFF/Image_{img_index}/PSNR', current_psnr, l)
                writer.add_image(f'REDDIFF/Image_{img_index}/Progress', (mu[0].clamp(-1, 1) + 1)/2, l)
    
    # 保存最终图像
    os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'input'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)
    
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(mu))
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    # 计算最终指标
    final_img_np = mu.cpu().squeeze().detach().numpy()
    if final_img_np.shape[0] == 3:
        final_img_np = np.transpose(final_img_np, (1, 2, 0))
    
    final_psnr = peak_signal_noise_ratio(ref_img_np, final_img_np)
    final_ssim = structural_similarity(ref_img_np, final_img_np, channel_axis=2, data_range=1)
    final_lpips = loss_fn_alex(ref_img_torch, current_img_torch).item()
    
    # 记录最终指标
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    print(f"REDDIFF Final Metrics - PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return mu, final_metric