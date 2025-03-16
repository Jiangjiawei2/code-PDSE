# util/algo/dmplug.py
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

def DMPlug(
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
    DMPlug算法：使用扩散模型作为先验，通过迭代优化重建图像。
    """
    # 设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # TensorBoard记录超参数和初始状态
    if writer is not None and img_index is not None:
        writer.add_text(f'DMPlug/Image_{img_index}/Config', 
                       f'Iterations: {iteration}\n'
                       f'Learning Rate: {lr}\n'
                       f'Denoiser Steps: {denoiser_step}\n'
                       f'Early Stopping: threshold={early_stopping_threshold}, patience={stop_patience}\n'
                       f'Random Seed: {random_seed}', 0)
        
        # 记录参考图像和测量图像
        writer.add_image(f'DMPlug/Image_{img_index}/Reference', (ref_img[0] + 1)/2, 0)
        writer.add_image(f'DMPlug/Image_{img_index}/Measurement', (y_n[0] + 1)/2, 0)
    
    # 初始化变量
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}])
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    
    # 记录初始随机噪声
    if writer is not None and img_index is not None:
        writer.add_image(f'DMPlug/Image_{img_index}/Initial_Noise', (Z[0] + 1)/2, 0)

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
    for epoch in tqdm(range(iteration), desc="DMPlug Optimization"):
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
            
            # TensorBoard记录
            if writer is not None and img_index is not None:
                metrics_dict = {
                    'Training/Loss': loss.item(),
                    'Metrics/PSNR': current_psnr,
                    'Metrics/SSIM': current_ssim,
                    'Metrics/LPIPS': current_lpips
                }
                log_metrics_to_tensorboard(writer, metrics_dict, epoch, img_index, prefix='DMPlug/Image')
                
                # 每隔一定轮数记录中间过程图像
                if epoch % 100 == 0 or epoch == iteration - 1:
                    writer.add_image(f'DMPlug/Image_{img_index}/Intermediate/Epoch_{epoch}', 
                                   (sample[0] + 1)/2, epoch)
                
                # 只记录最佳PSNR的中间结果
                if current_psnr == best_psnr:
                    writer.add_scalar(f'DMPlug/Image_{img_index}/Best/PSNR', best_psnr, epoch)
        
        # 早停检查
        if epoch > stop_patience:
            recent_changes = mean_changes[-stop_patience:]
            if all(abs(recent_changes[i] - recent_changes[i-1]) < early_stopping_threshold 
                  for i in range(1, len(recent_changes))):
                print(f"Early stopping triggered at epoch {epoch+1}")
                if writer is not None and img_index is not None:
                    writer.add_text(f'DMPlug/Image_{img_index}/EarlyStopping', 
                                   f'Stopped at epoch {epoch+1} due to stability in image mean', 0)
                break
    
    # 如果没有找到最佳图像，使用最后一个
    if best_img is None:
        best_img = sample
    
    # 记录训练曲线
    if writer is not None and img_index is not None:
        plot_and_log_curves(
            writer=writer,
            losses=losses,
            psnrs=psnrs,
            ssims=ssims,
            lpipss=lpipss,
            out_path=out_path,
            img_index=img_index,
            algo_name="DMPlug"
        )    
    # 保存最终图像
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
    
    # 记录最终指标
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 记录到TensorBoard
    if writer is not None and img_index is not None:
        writer.add_scalar(f'DMPlug/Image_{img_index}/Final/PSNR', final_psnr, 0)
        writer.add_scalar(f'DMPlug/Image_{img_index}/Final/SSIM', final_ssim, 0)
        writer.add_scalar(f'DMPlug/Image_{img_index}/Final/LPIPS', final_lpips, 0)
    
    print(f"Final metrics between reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return best_img, final_metric


def DMPlug_turbulence(
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
    task_config,
    fname,
    kernel_ref,
    early_stopping_threshold=0.01,  # 统计特性变化的阈值
    stop_patience=5,  
    out_path="outputs",
    iteration=2000,
    lr=0.02,
    denoiser_step=3,
    mask=None,
    random_seed=None
):
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
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    trainable_kernel = torch.randn((1, kernel_size * kernel_size), device=device, requires_grad=True)
    trainable_tilt = torch.randn(1, 2, 256, 256, device=device) * 0.01
    trainable_tilt.requires_grad = True
    
    criterion = torch.nn.MSELoss().to(device)
    params_group1 = {'params': Z, 'lr': lr}
    params_group2 = {'params': trainable_kernel, 'lr': lrk}
    params_group3 = {'params': trainable_tilt, 'lr': lrt}
    optimizer = torch.optim.Adam([params_group1,params_group2,params_group3])

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
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
                sample, pred_start = sampler.p_sample(model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
            else:
                sample, pred_start = sampler.p_sample(model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
        
        sample = torch.clamp(sample, -1, 1)
        kernel_output = F.softmax(trainable_kernel, dim=1)
        out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
        loss = criterion(operator.forward(sample, out_k, trainable_tilt), y_n)

        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            metrics = compute_metrics(
            sample=sample,
            ref_img=ref_img,
            out_path=out_path,
            device=device,
            loss_fn_alex=loss_fn_alex,
            metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
        )

    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
    plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(out_k))
    plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel_ref))

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

     # 将结果组织到字典中
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, final_metric