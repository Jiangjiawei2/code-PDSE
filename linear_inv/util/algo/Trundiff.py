# util/algo/red_diff.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard, EarlyStopping, ESWithWMV
def RED_diff(
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
    random_seed=None
):
    """
    RED_diff: 使用扩散模型和RED框架实现逆问题求解的算法
    """
    # 使用传入的随机种子重新设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    
    # 初始化待优化的变量 x
    # x = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)

    alpha = torch.tensor(0.5, requires_grad=True, device=device)

    # optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.Adam([Z], lr=0.02)
    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}, {'params': alpha, 'lr': lr}])    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()

    psnrs, ssims, losses, lpipss, recent_psnrs = [], [], [], [], []
    mean_changes, var_changes = [], []  # 记录均值和方差的变化
    
    # Training loop
    for epoch in tqdm(range(iteration), desc="Training Epochs"):
        model.eval()
        optimizer.zero_grad()

        for i, t in enumerate(list(range(denoiser_step))[::-1]):
            time = torch.tensor([t] * ref_img.shape[0], device=device)
            if i == 0:
                sample, pred_start = sampler.p_sample(model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask)
            else:
                sample, pred_start = sampler.p_sample(model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask)
        
        difference = y_n - operator.forward(sample)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=sample)[0]
            
        v_k = sample - norm_grad
        
        x_k = alpha*sample + (1-alpha) * v_k
        
        # x = sample
        # 数据一致性项 0.5 * ||Ax - y||^2
                # Loss calculation
        if mask is not None:
        # loss = criterion(operator.forward(x_t), y_n)
            data_fidelity = l1_loss(operator.forward(x_k, mask=mask), y_n)
        else:            
            data_fidelity = l1_loss(operator.forward(x_k), y_n)
        
                # 总的目标函数
        loss = data_fidelity
        


        loss.backward(retain_graph=True)
        # loss.backward()
        
        optimizer.step()
        losses.append(loss.item())
        
        with torch.no_grad():
            metrics = compute_metrics(
            sample=x_k,
            ref_img=ref_img,
            out_path=out_path,
            device=device,
            loss_fn_alex=loss_fn_alex,
            metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
        )

    ## 保存最终图像
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    # 转换最佳图像和参考图像为 numpy 格式
    best_img_np = x_k.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
    ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

    # 计算 PSNR
    final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
    # 计算 SSIM
    final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
    # 计算 LPIPS
    best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

     # 将结果组织到字典中
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return x_k, final_metric



def acce_RED_diff(   
    model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config,
    measure_config, operator, fname, iter_step=3, iteration=1000, denoiser_step=10, 
    stop_patience=5, early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', 
    mask=None, random_seed=None, writer=None, img_index=None
):
    """
    acce_RED_diff算法: 加速版的RED (Regularization by Denoising) 使用扩散模型
    """
     # 首先导入所有必需的模块
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torchvision
    from tqdm import tqdm
    # 使用传入的随机种子重新设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
    
    # 记录配置到TensorBoard
    if writer is not None and img_index is not None:
        writer.add_text(f'acce_RED_diff/Image_{img_index}/Config', 
                       f'Iterations: {iteration}\n'
                       f'Learning Rate: {lr}\n'
                       f'Iter Step: {iter_step}\n'
                       f'Denoiser Steps: {denoiser_step}\n'
                       f'Early Stopping: threshold={early_stopping_threshold}, patience={stop_patience}\n'
                       f'Random Seed: {random_seed}', 0)
        
        # 记录参考图像和测量图像
        if ref_img.dim() == 4:  # [B, C, H, W]
            ref_to_log = (ref_img[0] + 1) / 2  # 规范化到[0,1]
        else:  # [C, H, W]
            ref_to_log = (ref_img + 1) / 2
        writer.add_image(f'acce_RED_diff/Image_{img_index}/Reference', ref_to_log, 0)
        
        if y_n.dim() == 4:  # [B, C, H, W]
            y_to_log = (y_n[0] + 1) / 2  # 规范化到[0,1]
        else:  # [C, H, W]
            y_to_log = (y_n + 1) / 2
        writer.add_image(f'acce_RED_diff/Image_{img_index}/Measurement', y_to_log, 0)
    
    # 初始化
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    
    # 记录初始随机噪声
    if writer is not None and img_index is not None:
        writer.add_image(f'acce_RED_diff/Image_{img_index}/Initial_Noise', (Z[0] + 1) / 2, 0)
    
    # 初始化损失函数和优化器
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    
    # 设定需要优化的参数
    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    mu = torch.tensor(1.0, requires_grad=True, device=device)

    # 记录指标和损失
    losses = []
    psnrs = []
    ssims = []
    lpipss = []

    # 初始采样过程
    sample = Z
    with torch.no_grad():  # 禁用梯度计算
        for i, t in enumerate(tqdm(list(range(denoiser_step))[::-1], desc="初始去噪步骤")):
            time = torch.tensor([t] * (1 if ref_img.dim() == 3 else ref_img.shape[0]), device=device)
            if i >= denoiser_step - iter_step:
                print(f"在最后{iter_step}步停止（步骤{i+1}）")
                break
            if i == 0:
                sample, pred_start = sampler.p_sample(
                    model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
                )
            else:
                sample, pred_start = sampler.p_sample(
                    model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
                )
        
        # 初始采样后记录指标
        if writer is not None and img_index is not None:
            writer.add_image(f'acce_RED_diff/Image_{img_index}/Initial_Sample', (sample[0] + 1) / 2, 0)

    # 优化过程
    sample = sample.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([
        {'params': sample, 'lr': lr}, 
        {'params': alpha, 'lr': lr}
    ])
    
    # 初始化新的早停器 (加权移动方差早停) - 调整为更保守的配置
    early_stopper = ESWithWMV(
        window_size=8,                        
        var_threshold=0.002, 
        alpha=0.01,     
        patience=8,               
        min_epochs=30,                        
        verbose=True                         
    )
    
    best_loss = float('inf')
    best_sample = None
    best_metrics = None
    best_epoch = 0
    
    pbar = tqdm(range(iteration), desc="acce_RED_diff优化")
    for epoch in pbar:
        model.eval()
        optimizer.zero_grad()
        x_t = sample
        x_t.requires_grad_(True)
        
        # 更新x_t
        for i, t in enumerate(list(range(iter_step))[::-1]):
            time = torch.tensor([t] * (1 if ref_img.dim() == 3 else ref_img.shape[0]), device=device)
            x_t, pred_start = sampler.p_sample(
                model=model, x=x_t, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
            )
        
        # 计算数据保真项的梯度
        try:
            if mask is not None:
                difference = y_n - operator.forward(x_t, mask=mask)
            else:
                difference = y_n - operator.forward(x_t)

            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0]
            
            # 计算RED更新
            v_k = x_t - mu * norm_grad
            
            # 混合更新
            x_k = alpha * x_t + (1 - alpha) * v_k
            
            # 计算损失
            if mask is not None:
                loss = l1_loss(operator.forward(x_k, mask=mask), y_n)
            else:
                loss = l1_loss(operator.forward(x_k), y_n)
            
            losses.append(loss.item())
            
            # 反向传播和优化步骤
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item(), 'alpha': alpha.item()})
            
            # 计算评价指标
            with torch.no_grad():
                # 确保ref_img和x_k维度一致
                if ref_img.dim() == 4 and x_k.dim() == 3:
                    x_k_compare = x_k.unsqueeze(0)
                else:
                    x_k_compare = x_k
                
                # 转换为numpy计算指标
                x_k_np = x_k_compare.detach().cpu().squeeze().numpy()
                ref_np = ref_img.detach().cpu().squeeze().numpy()
                
                # 确保是[H, W, C]格式
                if x_k_np.shape[0] == 3 and len(x_k_np.shape) == 3:
                    x_k_np = np.transpose(x_k_np, (1, 2, 0))
                if ref_np.shape[0] == 3 and len(ref_np.shape) == 3:
                    ref_np = np.transpose(ref_np, (1, 2, 0))
                
                # 计算PSNR
                try:
                    from skimage.metrics import peak_signal_noise_ratio
                    current_psnr = peak_signal_noise_ratio(ref_np, x_k_np)
                    psnrs.append(current_psnr)
                except Exception as e:
                    print(f"PSNR计算错误: {e}")
                    current_psnr = 0
                    psnrs.append(current_psnr)
                
                # 计算SSIM
                try:
                    from skimage.metrics import structural_similarity
                    current_ssim = structural_similarity(ref_np, x_k_np, channel_axis=2, data_range=1)
                    ssims.append(current_ssim)
                except Exception as e:
                    print(f"SSIM计算错误: {e}, 尝试使用channel_axis参数")
                    try:
                        current_ssim = structural_similarity(ref_np, x_k_np, channel_axis=2, data_range=1)
                        ssims.append(current_ssim)
                    except Exception as e2:
                        print(f"SSIM计算仍然失败: {e2}")
                        current_ssim = 0
                        ssims.append(current_ssim)
                
                # 计算LPIPS
                try:
                    # 为LPIPS准备张量
                    x_k_lpips = torch.from_numpy(x_k_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    ref_lpips = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    current_lpips = loss_fn_alex(x_k_lpips, ref_lpips).item()
                    lpipss.append(current_lpips)
                except Exception as e:
                    print(f"LPIPS计算错误: {e}")
                    current_lpips = 0
                    lpipss.append(current_lpips)
                
                # 记录指标到TensorBoard
                if writer is not None and img_index is not None:
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/Loss', loss.item(), epoch)
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/PSNR', current_psnr, epoch)
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/SSIM', current_ssim, epoch)
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/LPIPS', current_lpips, epoch)
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/Alpha', alpha.item(), epoch)
                    
                    # 每隔10轮记录一次中间过程图像
                    if epoch % 10 == 0:
                        writer.add_image(f'acce_RED_diff/Image_{img_index}/Intermediate/Epoch_{epoch}', 
                                       (x_k[0] if x_k.dim() == 4 else x_k + 1) / 2, epoch)
                
                if early_stopper(epoch, x_k, loss.item()):
                    if epoch < 30 and len(losses) > 5:  
                        recent_losses = losses[-5:]
                        loss_trend = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
                        
                        if loss_trend > 0.1: 
                            print(f"早停被触发，但损失仍在显著下降({loss_trend:.2%})，继续训练")
                            early_stopper.counter = early_stopper.patience // 4  # 从1/2减少到1/4
                            continue
                    
                    print(f"早停触发，在第{epoch+1}轮停止训练")
                    break
                
                # 保存最佳样本
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_sample = x_k.clone().detach()
                    best_metrics = {
                        'psnr': current_psnr,
                        'ssim': current_ssim,
                        'lpips': current_lpips
                    }
                    best_epoch = epoch
                    
                    # 记录最佳样本对应的指标
                    if writer is not None and img_index is not None:
                        writer.add_text(f'acce_RED_diff/Image_{img_index}/Best/Info', 
                                       f'Epoch: {best_epoch}\n'
                                       f'Loss: {best_loss:.6f}\n'
                                       f'PSNR: {current_psnr:.4f}\n'
                                       f'SSIM: {current_ssim:.4f}\n'
                                       f'LPIPS: {current_lpips:.4f}', best_epoch)
            
        except Exception as e:
            print(f"优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 如果没有找到最佳样本，使用最后一个样本
    if best_sample is None:
        best_sample = x_k
        best_metrics = {
            'psnr': psnrs[-1] if psnrs else 0,
            'ssim': ssims[-1] if ssims else 0,
            'lpips': lpipss[-1] if lpipss else 0
        }
    
        # 保存结果曲线 - 仅使用TensorBoard
    try:
        if losses and writer is not None and img_index is not None:
            
            # 直接向TensorBoard添加标量数据
            for i, loss in enumerate(losses):
                writer.add_scalar(f'acce_RED_diff/Image_{img_index}/Loss', loss, i)
            
            # 添加PSNR数据
            if psnrs:
                for i, psnr in enumerate(psnrs):
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/PSNR', psnr, i)
            
            # 添加SSIM数据
            if ssims:
                for i, ssim in enumerate(ssims):
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/SSIM', ssim, i)
            
            # 添加LPIPS数据
            if lpipss:
                for i, lpips_val in enumerate(lpipss):
                    writer.add_scalar(f'acce_RED_diff/Image_{img_index}/LPIPS', lpips_val, i)
            print(f"成功将学习曲线数据添加到TensorBoard")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"记录结果曲线到TensorBoard时出错: {e}")
    
        # 保存重建图像到文件系统 - 优化版
    try:
        # 确保所有必要的导入
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端，避免显示问题
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # 确保目录存在 - 使用多种方式尝试创建目录
        try:
            for subdir in ['recon', 'input', 'label']:
                # 方式1: os.path.join
                full_path = os.path.join(out_path, subdir)
                os.makedirs(full_path, exist_ok=True)
        except Exception:
            # 方式2: 字符串拼接
            for subdir in ['recon', 'input', 'label']:
                full_path = out_path + '/' + subdir
                os.makedirs(full_path, exist_ok=True)
        
        # 使用安全的图像保存函数
        def save_tensor_image(tensor, path):
            """保存张量图像到文件 - 增强版"""
            try:
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
                
                # 方法1: 使用matplotlib保存
                plt.imsave(path, img_np)
                plt.close()  # 确保关闭图形
                
                # 验证文件是否成功创建
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise Exception("文件未成功创建或为空")
                    
                return True
            except Exception as e:
                print(f"使用matplotlib保存图像失败: {e}")
                
                try:
                    # 方法2: 使用PIL保存
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                    img_pil.save(path)
                    return True
                except Exception as e2:
                    print(f"使用PIL保存图像也失败: {e2}")
                    return False
        
        # 准备文件路径
        try:
            # 方式1: 使用os.path.join
            recon_path = os.path.join(out_path, 'recon', fname)
            input_path = os.path.join(out_path, 'input', fname)
            label_path = os.path.join(out_path, 'label', fname)
        except Exception:
            # 方式2: 使用字符串拼接
            recon_path = out_path + '/recon/' + fname
            input_path = out_path + '/input/' + fname
            label_path = out_path + '/label/' + fname
        
        # 保存图像
        recon_saved = save_tensor_image(best_sample, recon_path)
        input_saved = save_tensor_image(y_n, input_path)
        label_saved = save_tensor_image(ref_img, label_path)
        
        if recon_saved and input_saved and label_saved:
            print(f"成功保存所有图像到 {out_path}")
        else:
            print("部分图像保存失败，尝试备选方案")
            raise Exception("需要使用备选方案")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"保存图像时出错: {e}，尝试备选方案")
        
        # 备选方案
        try:
            import os
            import matplotlib.pyplot as plt
            
            # 确保目录存在
            for subdir in ['recon', 'input', 'label']:
                full_path = out_path + '/' + subdir
                os.makedirs(full_path, exist_ok=True)
            
            # 使用clear_color函数处理
            if 'clear_color' in globals():
                plt.imsave(out_path + '/recon/' + fname, clear_color(best_sample))
                plt.imsave(out_path + '/input/' + fname, clear_color(y_n))
                plt.imsave(out_path + '/label/' + fname, clear_color(ref_img))
                plt.close()
                print("使用备选方案成功保存图像")
            else:
                # 如果clear_color不可用，尝试最简单的保存方法
                def simple_save(tensor, path):
                    # 转换为numpy并简单缩放
                    img = tensor.detach().cpu().squeeze().numpy()
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    plt.imsave(path, img)
                    
                simple_save(best_sample, out_path + '/recon/' + fname)
                simple_save(y_n, out_path + '/input/' + fname)
                simple_save(ref_img, out_path + '/label/' + fname)
                plt.close()
                print("使用简化方法成功保存图像")
                
        except Exception as e2:
            traceback.print_exc()
            print(f"所有保存方法均失败: {e2}")
    # 返回训练曲线数据以便进行统计分析
    psnr_curve = {
        'psnrs': psnrs,
    }
    
    # 返回最佳样本和指标
    return best_sample, best_metrics,psnr_curve




# 辅助函数：计算评价指标
def compute_metrics(
    sample, ref_img, out_path, device, loss_fn_alex, epoch=None, iteration=None, metrics=None
):
    """
    计算PSNR、SSIM和LPIPS指标。
    """
    # 初始化指标列表（如果未提供）
    if metrics is None:
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

    # 确保样本和参考图像具有正确的维度
    if sample.dim() == 3:  # 如果样本是[C, H, W]
        sample = sample.unsqueeze(0)  # 转换为[1, C, H, W]
    
    if ref_img.dim() == 3:  # 如果参考图像是[C, H, W]
        ref_img = ref_img.unsqueeze(0)  # 转换为[1, C, H, W]
    
    # 转换为numpy格式计算指标
    ref_numpy = ref_img.cpu().squeeze().numpy()
    output_numpy = sample.detach().cpu().squeeze().numpy()
    
    # 检查输出是否仍然是[C, H, W]格式
    if output_numpy.ndim == 3 and output_numpy.shape[0] == 3:
        output_numpy = output_numpy.transpose(1, 2, 0)  # 转换为[H, W, C]
    
    # 准备用于计算指标的数据
    if ref_numpy.ndim == 3 and ref_numpy.shape[0] == 3:
        ref_numpy_cal = ref_numpy.transpose(1, 2, 0)  # 转换为[H, W, C]
    else:
        ref_numpy_cal = ref_numpy

    # 计算PSNR
    try:
        from skimage.metrics import peak_signal_noise_ratio
        tmp_psnr = peak_signal_noise_ratio(ref_numpy_cal, output_numpy)
        metrics['psnr'].append(tmp_psnr)
    except Exception as e:
        print(f"PSNR计算错误: {e}")
        tmp_psnr = 0
        metrics['psnr'].append(tmp_psnr)

    # 计算SSIM
    try:
        from skimage.metrics import structural_similarity
        tmp_ssim = structural_similarity(ref_numpy_cal, output_numpy, channel_axis=2, data_range=1)
        metrics['ssim'].append(tmp_ssim)
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        tmp_ssim = 0
        metrics['ssim'].append(tmp_ssim)

    # 计算LPIPS
    try:
        import torch
        # 为LPIPS准备正确的张量
        rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        gt_img_torch = torch.from_numpy(ref_numpy_cal).permute(2, 0, 1).unsqueeze(0).float().to(device)
        lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
        metrics['lpips'].append(lpips_alex)
    except Exception as e:
        print(f"LPIPS计算错误: {e}")
        lpips_alex = 0
        metrics['lpips'].append(lpips_alex)

    # 确保输出目录存在
    import os
    os.makedirs(out_path, exist_ok=True)
    
    # 将结果写入CSV文件
    file_path = os.path.join(out_path, "metrics_curve.csv")
    # 如果文件不存在或epoch是第一个epoch，则写入标题行
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    import csv
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "PSNR", "SSIM", "LPIPS"])
        writer.writerow([epoch if epoch is not None else len(metrics['psnr']), tmp_psnr, tmp_ssim, lpips_alex])
    
    # 在最后一个epoch后清空文件以便下一个图像使用
    if epoch is not None and iteration is not None and epoch == iteration - 1:
        open(file_path, 'w').close()  # 清空文件
    
    return metrics


def acce_RED_diff_turbulence(
    model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config,
    measure_config, task_config, operator, fname, kernel_ref,
    iter_step=4, iteration=1000, denoiser_step=10, stop_patience=5, 
    early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None
):
    """
    acce_RED_diff_turbulence: 针对大气湍流退化的加速RED算法
    """
    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Initialize variables and tensors
    kernel_type = task_config["kernel"]
    kernel_size = task_config["kernel_size"]
    intensity = task_config["intensity"]

    lrk= 1e-1
    lrt = 1e-7
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    trainable_kernel = torch.randn((1, kernel_size * kernel_size), device=device, requires_grad=True)
    trainable_tilt = torch.randn(1, 2, 256, 256, device=device) * 0.01
    trainable_tilt.requires_grad = True
    
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    losses = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    mu = torch.tensor(1.0, requires_grad=True, device=device)

    # 初始采样过程
    with torch.no_grad():  # 禁用梯度计算
        for i, t in enumerate(tqdm(list(range(denoiser_step))[::-1], desc="Denoising Steps")):
            time = torch.tensor([t] * ref_img.shape[0], device=device)
            if i >= denoiser_step - iter_step:
                print(f"Stopping at the last {iter_step} steps (step {i+1})")
                break
            if i == 0:
                sample, pred_start = sampler.p_sample(
                    model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
                )
            else:
                sample, pred_start = sampler.p_sample(
                    model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
                )
                
            metrics = compute_metrics(
                sample=sample,
                ref_img=ref_img,
                out_path=out_path,
                device=device,
                loss_fn_alex=loss_fn_alex,
                metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
            )

    # 优化过程
    sample = sample.detach().clone().requires_grad_(True)

    params_group1 = {'params': sample, 'lr': lr}
    params_group2 = {'params': trainable_kernel, 'lr': lrk}
    params_group3 = {'params': trainable_tilt, 'lr': lrt}
    optimizer = torch.optim.Adam([params_group1,params_group2,params_group3])

    for epoch in tqdm(range(iteration), desc="Training Epochs"):
        model.eval()
        optimizer.zero_grad()
        x_t = sample
        x_t.requires_grad_(True)
        # 更新 x_t
        for i, t in enumerate(list(range(iter_step))[::-1]):
            time = torch.tensor([t] * ref_img.shape[0], device=device)
            x_t, pred_start = sampler.p_sample(
                model=model, x=x_t, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
            )
        
        # loss = criterion(operator.forward(sample, out_k, trainable_tilt), y_n)

        x_t = torch.clamp(x_t, -1, 1)
        kernel_output = torch.nn.functional.softmax(trainable_kernel, dim=1)
        out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
        
        difference = y_n - operator.forward(x_t, out_k, trainable_tilt)

        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0]
        
        v_k = x_t - mu * norm_grad
        
        x_k = alpha*x_t + (1-alpha) * v_k
        
        # 计算损失并优化        
        loss = l1_loss(operator.forward(x_k, out_k, trainable_tilt), y_n)

        loss.backward(retain_graph=True)

        optimizer.step()
        losses.append(loss.item())

        # 计算并记录指标
        with torch.no_grad():
            metrics = compute_metrics(
                sample=x_k,
                ref_img=ref_img,
                out_path=out_path,
                device=device,
                loss_fn_alex=loss_fn_alex,
                metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
            )

    # Save results
    Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(out_k))
    plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel_ref))

    
    # 转换最佳图像和参考图像为 numpy 格式
    best_img_np = x_t.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
    ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

    # 计算 PSNR
    final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
    # 计算 SSIM
    final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
    # 计算 LPIPS
    best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

    # 将结果组织到字典中
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return x_k, final_metric

