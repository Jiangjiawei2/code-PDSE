# util/algo/reddiff.py
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips
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
    stop_patience=10,
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
    reddif算法：使用扩散模型作为先验，通过迭代优化重建图像。
    """
    try:
        # 设置随机种子
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # TensorBoard记录超参数和初始状态
        if writer is not None and img_index is not None:
            writer.add_text(f'reddiff/Image_{img_index}/Config', 
                           f'Iterations: {iteration}\n'
                           f'Learning Rate: {lr}\n'
                           f'Denoiser Steps: {denoiser_step}\n'
                           f'Early Stopping: threshold={early_stopping_threshold}, patience={stop_patience}\n'
                           f'Random Seed: {random_seed}', 0)
            
            # 记录参考图像和测量图像
            writer.add_image(f'reddiff/Image_{img_index}/Reference', (ref_img[0] + 1)/2, 0)
            writer.add_image(f'reddiff/Image_{img_index}/Measurement', (y_n[0] + 1)/2, 0)
        
        # 初始化变量
        Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}])
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        criterion = torch.nn.MSELoss().to(device)
        l1_loss = torch.nn.L1Loss()
        
        # 记录初始随机噪声
        if writer is not None and img_index is not None:
            writer.add_image(f'reddiff/Image_{img_index}/Initial_Noise', (Z[0] + 1)/2, 0)

        # 用于记录指标和优化过程
        losses = []
        psnrs = []
        ssims = []
        lpipss = []
        mean_changes = []  # 记录均值变化
        best_psnr = 0
        best_img = None
        best_epoch = 0
        
        # 主优化循环
        pbar = tqdm(range(iteration), desc="reddiff Optimization")
        for epoch in pbar:
            model.eval()
            optimizer.zero_grad()
            
            # 应用扩散模型的去噪过程
            sample = Z
            for i, t in enumerate(list(range(denoiser_step))[::-1]):
                time = torch.tensor([t] * ref_img.shape[0], device=device)
                if i == 0:
                    sample, pred_start = sampler.p_sample(
                        model=model, x=sample, t=time, measurement=y_n, 
                        measurement_cond_fn=measurement_cond_fn, mask=mask
                    )
                else:
                    sample, pred_start = sampler.p_sample(
                        model=model, x=sample, t=time, measurement=y_n,
                        measurement_cond_fn=measurement_cond_fn, mask=mask
                    )
            
            # 计算损失
            if mask is not None:
                loss = criterion(operator.forward(sample, mask=mask), y_n)
            else:
                loss = criterion(operator.forward(sample), y_n)
            
            # 反向传播和优化
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
            
            # 计算当前重建的评价指标
            with torch.no_grad():
                metrics_result = compute_metrics(
                    sample=sample,
                    ref_img=ref_img,
                    out_path=out_path,
                    device=device,
                    loss_fn_alex=loss_fn_alex,
                    epoch=epoch
                )
                
                # 获取当前指标
                current_psnr = metrics_result['psnr'][-1]
                current_ssim = metrics_result['ssim'][-1]
                current_lpips = metrics_result['lpips'][-1]
                
                # 更新进度条显示当前loss和psnr
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{current_psnr:.2f}'
                })
                
                # 追加到记录列表
                psnrs.append(current_psnr)
                ssims.append(current_ssim)
                lpipss.append(current_lpips)
                current_img_np = sample.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
                
                # 记录图像均值变化，用于早停
                mean_val = np.mean(current_img_np)
                mean_changes.append(mean_val)
                
                # 记录最佳PSNR图像
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_img = sample.clone()
                    best_epoch = epoch
                
                # TensorBoard记录 - 简化版本，避免曲线图像问题
                if writer is not None and img_index is not None:
                    try:
                        # 只记录标量指标
                        writer.add_scalar(f'reddiff/Image_{img_index}/Loss', loss.item(), epoch)
                        writer.add_scalar(f'reddiff/Image_{img_index}/PSNR', current_psnr, epoch)
                        writer.add_scalar(f'reddiff/Image_{img_index}/SSIM', current_ssim, epoch)
                        writer.add_scalar(f'reddiff/Image_{img_index}/LPIPS', current_lpips, epoch)
                        
                        # 每隔一定轮数记录图像
                        if epoch % 100 == 0:
                            writer.add_image(f'reddiff/Image_{img_index}/Progress', 
                                           (sample[0].clamp(-1, 1) + 1)/2, epoch)
                    except Exception as e:
                        print(f"Warning: TensorBoard logging failed: {e}")
                        continue
            
            # # 早停检查
            # if epoch > stop_patience:
            #     recent_changes = mean_changes[-stop_patience:]
            #     if all(abs(recent_changes[i] - recent_changes[i-1]) < early_stopping_threshold 
            #           for i in range(1, len(recent_changes))):
            #         print(f"Early stopping triggered at epoch {epoch+1}")
            #         if writer is not None and img_index is not None:
            #             writer.add_text(f'reddiff/Image_{img_index}/EarlyStopping', 
            #                            f'Stopped at epoch {epoch+1} due to stability in image mean', 0)
            #         break
        
        # 保存最终结果
        os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'input'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)
        
        # 使用最佳PSNR图像作为最终结果
        if best_img is None:
            best_img = sample
        
        # 保存图像
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(best_img))
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        
        # 计算最终指标
        best_img_np = best_img.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
        ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)
        
        final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
        final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
        
        best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()
        
        # 最终指标
        final_metric = {
            'psnr': final_psnr,
            'ssim': final_ssim,
            'lpips': final_lpips
        }
        
        print(f"Final metrics between reconstructed image and reference image:")
        print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
        
        # 保存曲线数据到CSV文件
        try:
            curves_dir = os.path.join(out_path, 'curves')
            os.makedirs(curves_dir, exist_ok=True)
            
            # 保存PSNR曲线数据
            with open(os.path.join(curves_dir, f'psnr_curve_img{img_index}.csv'), 'w') as f:
                f.write('Epoch,PSNR\n')
                for i, psnr in enumerate(psnrs):
                    f.write(f'{i},{psnr}\n')
                    
            # 保存SSIM曲线数据
            with open(os.path.join(curves_dir, f'ssim_curve_img{img_index}.csv'), 'w') as f:
                f.write('Epoch,SSIM\n')
                for i, ssim in enumerate(ssims):
                    f.write(f'{i},{ssim}\n')
                    
            # 保存LPIPS曲线数据
            with open(os.path.join(curves_dir, f'lpips_curve_img{img_index}.csv'), 'w') as f:
                f.write('Epoch,LPIPS\n')
                for i, lpips_val in enumerate(lpipss):
                    f.write(f'{i},{lpips_val}\n')
        except Exception as e:
            print(f"Warning: Failed to save curve data: {e}")
        
        return best_img, final_metric

    except Exception as e:
        print(f"Error in reddiff: {str(e)}")
        print(f"Input shapes - ref_img: {ref_img.shape}, y_n: {y_n.shape}")
        import traceback
        print(traceback.format_exc())
        raise e



# def DMPlug_turbulence(
#     model,
#     sampler,
#     measurement_cond_fn,
#     ref_img,
#     y_n,
#     args,
#     operator,
#     device,
#     model_config,
#     measure_config,
#     task_config,
#     fname,
#     kernel_ref,
#     early_stopping_threshold=0.01,  # 统计特性变化的阈值
#     stop_patience=5,  
#     out_path="outputs",
#     iteration=2000,
#     lr=0.02,
#     denoiser_step=3,
#     mask=None,
#     random_seed=None
# ):
#     # 使用传入的随机种子重新设置随机种子
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed)
#     # Initialize variables and tensors
#     kernel_type = task_config["kernel"]
#     kernel_size = task_config["kernel_size"]
#     intensity = task_config["intensity"]

#     lrk= 1e-1
#     lrt = 1e-7
#     Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
#     trainable_kernel = torch.randn((1, kernel_size * kernel_size), device=device, requires_grad=True)
#     trainable_tilt = torch.randn(1, 2, 256, 256, device=device) * 0.01
#     trainable_tilt.requires_grad = True
    
#     criterion = torch.nn.MSELoss().to(device)
#     params_group1 = {'params': Z, 'lr': lr}
#     params_group2 = {'params': trainable_kernel, 'lr': lrk}
#     params_group3 = {'params': trainable_tilt, 'lr': lrt}
#     optimizer = torch.optim.Adam([params_group1,params_group2,params_group3])

#     loss_fn_alex = lpips.LPIPS(net='alex').to(device)
#     l1_loss = torch.nn.L1Loss()

    
#     psnrs, ssims, losses, lpipss, recent_psnrs = [], [], [], [], []
#     mean_changes, var_changes = [], []  # 记录均值和方差的变化
    
#     # Training loop
#     for epoch in tqdm(range(iteration), desc="Training Epochs"):
#         model.eval()
#         optimizer.zero_grad()

        
#         for i, t in enumerate(list(range(denoiser_step))[::-1]):
#             time = torch.tensor([t] * ref_img.shape[0], device=device)
#             if i == 0:
#                 sample, pred_start = sampler.p_sample(model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
#             else:
#                 sample, pred_start = sampler.p_sample(model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
        
#         sample = torch.clamp(sample, -1, 1)
#         kernel_output = F.softmax(trainable_kernel, dim=1)
#         out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
#         loss = criterion(operator.forward(sample, out_k, trainable_tilt), y_n)

#         loss.backward(retain_graph=True)
#         optimizer.step()
#         losses.append(loss.item())

#         with torch.no_grad():
#             metrics = compute_metrics(
#             sample=sample,
#             ref_img=ref_img,
#             out_path=out_path,
#             device=device,
#             loss_fn_alex=loss_fn_alex,
#             metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
#         )

#     plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
#     plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(out_k))
#     plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel_ref))

#     best_img_np = sample.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
#     ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

#     # 计算 PSNR
#     final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
#     # 计算 SSIM
#     final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
#     # 计算 LPIPS
#     best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#     ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#     final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

#      # 将结果组织到字典中
#     final_metric = {
#         'psnr': final_psnr,
#         'ssim': final_ssim,
#         'lpips': final_lpips
#     }
    
#     # 打印最终的 PSNR, SSIM, LPIPS
#     print(f"Final metrics between best reconstructed image and reference image:")
#     print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
#     return sample, final_metric



# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import lpips
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from util.img_utils import clear_color

# def reddiff(
#     model,
#     sampler,
#     measurement_cond_fn,
#     ref_img,
#     y_n,
#     args,
#     operator,
#     device,
#     model_config,
#     measure_config,
#     fname,
#     out_path="outputs",
#     iteration=2000,
#     lr=0.02,
#     denoiser_step=3,
#     mask=None,
#     random_seed=None,
#     writer=None,
#     img_index=None
# ):
#     """
#     REDDIFF算法实现
    
#     基于Algorithm 1 Variational sampler (RED-diff)的实现:
#     Input: y, f_θ, σ_v, L, {α_t, σ_t, λ_t}_{t=1}^T
#     Initialize: μ_0
#     for l = 1, ..., L do
#         t ~ U[0, T]
#         ε ~ N(0, I_n)
#         x_t = α_t·μ + σ_t·ε
#         loss = ||y - f(μ)||^2 + λ_t·(sg[ε_θ(x_t; t) - ε])^T·μ
#         μ ← OptimizerStep(loss)
#     end for
#     Return: μ
    
#     参数:
#     - model: 扩散模型 f_θ
#     - sampler: 采样器
#     - measurement_cond_fn: 测量条件函数
#     - ref_img: 参考图像
#     - y_n: 带噪声的测量 y
#     - args: 参数集合
#     - operator: 前向操作符 f
#     - device: 运算设备
#     - model_config: 模型配置
#     - measure_config: 测量配置
#     - fname: 输出文件名
#     - out_path: 输出路径
#     - iteration: 最大迭代次数 L
#     - lr: 学习率
#     - denoiser_step: 扩散模型步数
#     - mask: 可选掩码
#     - random_seed: 随机种子
#     - writer: TensorBoard写入器
#     - img_index: 图像索引
    
#     返回:
#     - 重建图像 μ
#     - 评价指标字典
#     """
#     # 设置随机种子以确保结果可复现
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#         torch.cuda.manual_seed(random_seed)
#         torch.cuda.manual_seed_all(random_seed)
    
#     # 初始化评价指标计算工具
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device)
#     criterion = torch.nn.MSELoss().to(device)
    
#     # 初始化变量：μ是要优化的潜在变量
#     # mu = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
#     initx=operator.transpose(y_n)
#     mu = initx.clone().detach().requires_grad_(True)    # 初始化优化器
#     optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    
#     # 算法中λ_t参数 (噪声预测项权重)
#     grad_term_weight = getattr(args, 'grad_term_weight', 0.25)
    
#     # 用于记录指标
#     losses = []
#     psnrs = []
#     ssims = []
#     lpipss = []
    
#     # 记录初始状态到TensorBoard
#     if writer is not None and img_index is not None:
#         writer.add_image(f'REDDIFF/Image_{img_index}/Initial', (mu[0] + 1)/2, 0)
    
#     # 主优化循环 (for l = 1, ..., L)
#     pbar = tqdm(range(iteration), desc="REDDIFF Optimization")
#     for l in pbar:
#         model.eval()
#         optimizer.zero_grad()
        
#         # 随机采样时间步 t ~ U[0, T]
#         t = torch.randint(0, sampler.num_timesteps, (1,), device=device)[0]
#         time = torch.tensor([t] * mu.shape[0], device=device)
        
#         # 获取噪声调度参数
#         alpha_t = torch.tensor(sampler.alphas_cumprod[t], device=device).float()
#         sigma_t = torch.sqrt(1 - alpha_t)
        
#         # 采样噪声 ε ~ N(0, I_n)
#         epsilon = torch.randn_like(mu)
        
#         noise_x0 = torch.randn_like(mu)
        
#         x0_pred = mu + 0.0001*noise_x0
#         # 构造噪声数据点 x_t = α_t·μ + σ_t·ε
#         x_t = alpha_t.sqrt() * x0_pred + sigma_t * epsilon
        
#         # 通过扩散模型获取噪声预测
#         with torch.no_grad():
#             # 使用p_mean_variance获取预测
#             out = sampler.p_mean_variance(model, x_t, time)
#             pred_xstart = out['pred_xstart']
            
#             # 从x0预测推导噪声预测
#             et = sampler.predict_eps_from_x_start(x_t, time, pred_xstart)
        
#         # 计算数据一致性损失
#         if mask is not None:
#             data_fidelity = criterion(operator.forward(x0_pred, mask=mask), y_n)
#         else:
#             data_fidelity = criterion(operator.forward(x0_pred), y_n)
        
#         # 计算先验损失 λ_t·(sg[ε_θ(x_t; t) - ε])^T·μ
#         # sg[·]表示stop gradient，即et.detach()
#         prior_loss = torch.mul((et.detach() - epsilon), pred_xstart).mean()
        
#         # 信噪比相关的权重调整
#         snr = alpha_t / sigma_t
#         lambda_t = grad_term_weight / snr
        
#         # 总损失
#         loss = data_fidelity + lambda_t * prior_loss
        
#         # 反向传播和优化
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
        
#         # 计算当前重建的评价指标
#         with torch.no_grad():
#             current_img_np = mu.cpu().squeeze().detach().numpy()
#             if current_img_np.shape[0] == 3:  # 转换为[H, W, C]格式
#                 current_img_np = np.transpose(current_img_np, (1, 2, 0))
                
#             ref_img_np = ref_img.cpu().squeeze().numpy()
#             if ref_img_np.shape[0] == 3:
#                 ref_img_np = np.transpose(ref_img_np, (1, 2, 0))
            
#             # 计算PSNR
#             current_psnr = peak_signal_noise_ratio(ref_img_np, current_img_np)
#             psnrs.append(current_psnr)
            
#             # 计算SSIM
#             current_ssim = structural_similarity(ref_img_np, current_img_np, channel_axis=2, data_range=1)
#             ssims.append(current_ssim)
            
#             # 计算LPIPS
#             current_img_torch = torch.from_numpy(current_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#             ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#             current_lpips = loss_fn_alex(ref_img_torch, current_img_torch).item()
#             lpipss.append(current_lpips)
            
#             # 更新进度条信息
#             pbar.set_postfix({
#                 'Loss': f"{loss.item():.4f}",
#                 'PSNR': f"{current_psnr:.2f}", 
#                 'SSIM': f"{current_ssim:.4f}"
#             })
            
#             # TensorBoard记录
#             if writer is not None and img_index is not None and l % 50 == 0:
#                 writer.add_scalar(f'REDDIFF/Image_{img_index}/Loss', loss.item(), l)
#                 writer.add_scalar(f'REDDIFF/Image_{img_index}/PSNR', current_psnr, l)
#                 writer.add_image(f'REDDIFF/Image_{img_index}/Progress', (mu[0].clamp(-1, 1) + 1)/2, l)
    
#     # 保存最终图像
#     os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
#     os.makedirs(os.path.join(out_path, 'input'), exist_ok=True)
#     os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)
    
#     plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(mu))
#     plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
#     plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
#     # 计算最终指标
#     final_img_np = mu.cpu().squeeze().detach().numpy()
#     if final_img_np.shape[0] == 3:
#         final_img_np = np.transpose(final_img_np, (1, 2, 0))
    
#     final_psnr = peak_signal_noise_ratio(ref_img_np, final_img_np)
#     final_ssim = structural_similarity(ref_img_np, final_img_np, channel_axis=2, data_range=1)
#     final_lpips = loss_fn_alex(ref_img_torch, current_img_torch).item()
    
#     # 记录最终指标
#     final_metric = {
#         'psnr': final_psnr,
#         'ssim': final_ssim,
#         'lpips': final_lpips
#     }
    
#     print(f"REDDIFF Final Metrics - PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
#     return mu, final_metric