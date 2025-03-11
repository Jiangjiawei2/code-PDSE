import torch
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import lpips
from util.img_utils import clear_color, mask_generator, normalize_np, clear
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience=40, min_delta=0, verbose=False):
        """
        初始化早停策略
        
        参数：
        - patience: 等待多少个 epoch 后，如果验证指标不再改善，则停止训练。
        - min_delta: 允许的最小变化量，低于该变化量时不认为是改进。
        - verbose: 是否打印早停信息。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """
        调用时，检查当前的指标是否足够好来更新最优指标。
        """
        if self.best_score is None:
            self.best_score = metric
        elif metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0  # reset counter if metric improves
        else:
            self.counter += 1  # increase counter if no improvement

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")

    def stop_training(self):
        return self.early_stop

def log_metrics_to_tensorboard(writer, metrics, step, img_index=None, prefix=''):
    """将指标记录到TensorBoard
    
    参数:
        writer: TensorBoard SummaryWriter对象
        metrics: 包含指标值的字典
        step: 训练步骤
        img_index: 可选的图像索引，用于区分不同图像
        prefix: 指标名称前缀
    """
    # 为每个指标添加适当的前缀（如果提供）
    tag_prefix = f"{prefix}/" if prefix else ""
    
    # 如果提供了图像索引，在标签中加入图像索引
    if img_index is not None:
        for metric_name, value in metrics.items():
            writer.add_scalar(f"{tag_prefix}{metric_name}/image_{img_index}", value, step)
    else:
        # 没有图像索引，直接记录指标
        for metric_name, value in metrics.items():
            writer.add_scalar(f"{tag_prefix}{metric_name}", value, step)
            
def compute_metrics(
    sample, ref_img, out_path, device, loss_fn_alex, epoch=None, iteration=None, metrics=None
):
    """
    计算PSNR、SSIM和LPIPS指标，保存结果到CSV文件，并记录最佳图像。
    
    参数:
        sample (torch.Tensor): 当前的采样结果张量。
        ref_img (torch.Tensor): 参考图像张量。
        out_path (str): 指标CSV文件保存路径。
        device (torch.device): 设备，通常为'cuda'或'cpu'。
        loss_fn_alex (lpips.LPIPS): LPIPS计算的损失函数。
        epoch (int, optional): 当前的epoch，用于记录进度。
        iteration (int, optional): 总迭代次数，用于清空文件等操作。
        metrics (dict, optional): 包含指标列表的字典，若为None，则自动初始化。
    
    返回:
        dict: 更新后的包含PSNR、SSIM、LPIPS列表
    """
    # 初始化指标列表（如果未提供）
    if metrics is None:
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

    # 转换为numpy格式计算指标
    ref_numpy = ref_img.cpu().squeeze().numpy()
    output_numpy = sample.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    ref_numpy_cal = ref_numpy.transpose(1, 2, 0)

    # 计算PSNR
    tmp_psnr = peak_signal_noise_ratio(ref_numpy_cal, output_numpy)
    metrics['psnr'].append(tmp_psnr)

    # 计算SSIM
    tmp_ssim = structural_similarity(ref_numpy_cal, output_numpy, channel_axis=2, data_range=1)
    metrics['ssim'].append(tmp_ssim)

    # 计算LPIPS
    rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    gt_img_torch = torch.from_numpy(ref_numpy_cal).permute(2, 0, 1).unsqueeze(0).float().to(device)
    lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
    metrics['lpips'].append(lpips_alex)

    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)
    
    # 将结果写入CSV文件
    file_path = os.path.join(out_path, "metrics_curve.csv")
    # 如果文件不存在或epoch是第一个epoch，则写入标题行
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "PSNR", "SSIM", "LPIPS"])
        writer.writerow([epoch if epoch is not None else len(metrics['psnr']), tmp_psnr, tmp_ssim, lpips_alex])
    
    # 在最后一个epoch后清空文件以便下一个图像使用
    if epoch is not None and iteration is not None and epoch == iteration - 1:
        open(file_path, 'w').close()  # 清空文件
    
    return metrics



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
    
    Parameters:
    - model: 扩散模型
    - sampler: 扩散采样器
    - measurement_cond_fn: 测量条件函数
    - ref_img: 参考图像张量
    - y_n: 带噪声的测量张量
    - args: 命令行参数
    - operator: 测量算子
    - device: 设备
    - model_config: 模型配置
    - measure_config: 测量配置
    - fname: 输出文件名
    - early_stopping_threshold: 早停阈值
    - stop_patience: 早停耐心值
    - out_path: 输出路径
    - iteration: 迭代次数
    - lr: 学习率
    - denoiser_step: 去噪步数
    - mask: 掩码（可选）
    - random_seed: 随机种子
    - writer: TensorBoard SummaryWriter对象
    - img_index: 图像索引
    
    Returns:
    - sample: 重建的图像
    - metrics: 包含PSNR、SSIM和LPIPS的指标字典
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
        with torch.no_grad():
            current_img_np = sample.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
            ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)
            
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
                # 每个epoch记录损失和指标
                writer.add_scalar(f'DMPlug/Image_{img_index}/Training/Loss', loss.item(), epoch)
                writer.add_scalar(f'DMPlug/Image_{img_index}/Metrics/PSNR', current_psnr, epoch)
                writer.add_scalar(f'DMPlug/Image_{img_index}/Metrics/SSIM', current_ssim, epoch)
                writer.add_scalar(f'DMPlug/Image_{img_index}/Metrics/LPIPS', current_lpips, epoch)
                
                # 每隔一定轮数记录图像
                if epoch % 100 == 0 or epoch == iteration - 1:
                    writer.add_image(f'DMPlug/Image_{img_index}/Reconstruction/Epoch_{epoch}', 
                                   (sample[0] + 1)/2, epoch)
                
                # 记录最佳PSNR图像
                if current_psnr == best_psnr:
                    writer.add_image(f'DMPlug/Image_{img_index}/Best/Reconstruction', 
                                   (best_img[0] + 1)/2, epoch)
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
    
    # 保存结果和可视化
    if writer is not None and img_index is not None:
        # 绘制损失曲线
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axs[0, 0].plot(losses)
        axs[0, 0].set_title('Loss Curve')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        
        # PSNR曲线
        axs[0, 1].plot(psnrs)
        axs[0, 1].set_title('PSNR Curve')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('PSNR (dB)')
        
        # SSIM曲线
        axs[1, 0].plot(ssims)
        axs[1, 0].set_title('SSIM Curve')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('SSIM')
        
        # LPIPS曲线
        axs[1, 1].plot(lpipss)
        axs[1, 1].set_title('LPIPS Curve')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('LPIPS')
        
        plt.tight_layout()
        
        # 保存并添加到TensorBoard
        curves_path = os.path.join(out_path, f'dmplug_curves_{img_index}.png')
        plt.savefig(curves_path)
        plt.close()
        
        curves_img = plt.imread(curves_path)
        writer.add_image(f'DMPlug/Image_{img_index}/Curves', 
                       torch.from_numpy(curves_img).permute(2, 0, 1), 0)
        
        # 记录误差图
        error_map = torch.abs(ref_img - best_img)
        error_map = error_map / error_map.max()  # 归一化误差
        writer.add_image(f'DMPlug/Image_{img_index}/Error_Map', error_map[0], 0)
        
        # 记录最终状态
        writer.add_text(f'DMPlug/Image_{img_index}/Results', 
                       f'Best PSNR: {best_psnr:.4f} at epoch {best_epoch}\n'
                       f'Final Loss: {losses[-1]:.6f}\n', 0)
    
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

                
            # # Statistical characteristic-based early stopping
            # mean_val = np.mean(output_numpy)
            
            # if epoch > 0:  # 从第二轮开始计算变化率
            #     # 记录当前均值
            #     mean_changes.append(mean_val)

            #         # 如果变化率在一定窗口内小于阈值，则早停
            #     if len(mean_changes) >= stop_patience:
            #         recent_mean_changes = mean_changes[-stop_patience:]
            #         if all(change < early_stopping_threshold for change in recent_mean_changes):
            #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
            #             break
            # else:
            #     # 初始记录第一轮的均值
            #     mean_changes.append(mean_val)

    # Save results
    # Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    # plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
    plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(out_k))
    plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel_ref))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()
    
    # plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    # plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    
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

     # 将结果组织到字典中
    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample , final_metric





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
                sample, pred_start = sampler.p_sample(model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
            else:
                sample, pred_start = sampler.p_sample(model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask =mask)
        
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
            data_fidelity  = l1_loss(operator.forward(x_k, mask=mask), y_n)
        else:            
            data_fidelity  = l1_loss(operator.forward(x_k), y_n)
        
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

                
            # # Statistical characteristic-based early stopping
            # mean_val = np.mean(output_numpy)
            
            # if epoch > 0:  # 从第二轮开始计算变化率
            #     # 记录当前均值
            #     mean_changes.append(mean_val)

            #         # 如果变化率在一定窗口内小于阈值，则早停
            #     if len(mean_changes) >= stop_patience:
            #         recent_mean_changes = mean_changes[-stop_patience:]
            #         if all(change < early_stopping_threshold for change in recent_mean_changes):
            #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
            #             break
            # else:
            #     # 初始记录第一轮的均值
            #     mean_changes.append(mean_val)

    # Save results
    # Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    # plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()
    
    # plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    # plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    
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
    
    return x_k , final_metric



def acce_RED_diff(   ##  best performence
    model, sampler, measurement_cond_fn, ref_img, y_n , device, model_config,measure_config,operator,fname,
    iter_step=4, iteration=1000, denoiser_step=10, stop_patience=5, 
    early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None):

    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # 初始化
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    losses = []

    # alpha = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device,requires_grad=True)


    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    mu = torch.tensor(1.0, requires_grad=True, device=device)

    # optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.Adam([Z], lr=0.02)

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
                

    # 优化过程
    sample = sample.detach().clone().requires_grad_(True)
    # optimizer = torch.optim.Adam([sample], lr=lr)

    optimizer = torch.optim.Adam([{'params': sample, 'lr': lr}, {'params': alpha, 'lr': lr}])    

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
        
        if mask is not None:
            difference = y_n - operator.forward(x_t, mask=mask)
        else:
            difference = y_n - operator.forward(x_t)

        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0]
        
        # norm_log = torch.log(norm + 1)  # log(1 + norm) 确保不出现负数
        # mu = 10 *torch.sigmoid(norm_log)

        # # 使用幂次衰减
        # scale = 30  # 控制衰减的幅度
        # p = 1.5  # 控制衰减的速度，p 越大，衰减越快   幂次衰减
        # mu = 10 * (norm / (norm + scale)) ** p

        # mu = torch.sigmoid(norm)
        # Print mu in real-time
        # print(f"Current value of diff and mu:{norm.item()}, {mu.item()}")
        v_k = x_t - mu*  norm_grad
        
        x_k = alpha*x_t + (1-alpha) * v_k
        
        # 计算损失并优化
        if mask is not None:
        # loss = criterion(operator.forward(x_t), y_n)
            loss = l1_loss(operator.forward(x_k, mask=mask), y_n)
        else:            
            loss = l1_loss(operator.forward(x_k), y_n)

        # loss.backward()
        loss.backward(retain_graph=True)

        optimizer.step()
        losses.append(loss.item())

        # # 计算并记录指标
        with torch.no_grad():
            metrics = compute_metrics(
            sample=x_k,
            ref_img=ref_img,
            out_path=out_path,
            device=device,
            loss_fn_alex=loss_fn_alex,
            epoch=epoch,
            iteration=iteration,
            metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
        )

        # # 早停机制
        # mean_val = np.mean(output_numpy)
        # if epoch > 0:
        #     mean_changes.append(mean_val)
        #     if len(mean_changes) >= stop_patience:
        #         recent_mean_changes = mean_changes[-stop_patience:]
        #         if all(abs(recent_mean_changes[i] - recent_mean_changes[i-1]) < early_stopping_threshold for i in range(1, len(recent_mean_changes))):
        #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
        #             break
        # else:
        #     mean_changes.append(mean_val)

    # Save results
    # Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    # plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()
    
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
    
    return x_k , final_metric



def acce_RED_diff_turbulence(   ##  best performence
    model, sampler, measurement_cond_fn, ref_img, y_n , device, model_config,measure_config, task_config, operator,fname, kernel_ref,
    iter_step=4, iteration=1000, denoiser_step=10, stop_patience=5, 
    early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None):

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

    # alpha = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device,requires_grad=True)


    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    mu = torch.tensor(1.0, requires_grad=True, device=device)

    # optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.Adam([Z], lr=0.02)

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
    # optimizer = torch.optim.Adam([sample], lr=lr)

    params_group1 = {'params': sample, 'lr': lr}
    params_group2 = {'params': trainable_kernel, 'lr': lrk}
    params_group3 = {'params': trainable_tilt, 'lr': lrt}
    optimizer = torch.optim.Adam([params_group1,params_group2,params_group3])
    
    # optimizer = torch.optim.Adam([{'params': sample, 'lr': lr}, {'params': alpha, 'lr': lr}])    

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
        kernel_output = F.softmax(trainable_kernel, dim=1)
        out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
        
        difference = y_n - operator.forward(x_t, out_k, trainable_tilt)

        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0]
        
        # norm_log = torch.log(norm + 1)  # log(1 + norm) 确保不出现负数
        # mu = 10 *torch.sigmoid(norm_log)

        # # 使用幂次衰减
        # scale = 30  # 控制衰减的幅度
        # p = 1.5  # 控制衰减的速度，p 越大，衰减越快   幂次衰减
        # mu = 10 * (norm / (norm + scale)) ** p

        # mu = torch.sigmoid(norm)
        # Print mu in real-time
        # print(f"Current value of diff and mu:{norm.item()}, {mu.item()}")
        v_k = x_t - mu*  norm_grad
        
        x_k = alpha*x_t + (1-alpha) * v_k
        
        # 计算损失并优化        
        loss = l1_loss(operator.forward(x_k, out_k, trainable_tilt), y_n)

        # loss.backward()
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

        # # 早停机制
        # mean_val = np.mean(output_numpy)
        # if epoch > 0:
        #     mean_changes.append(mean_val)
        #     if len(mean_changes) >= stop_patience:
        #         recent_mean_changes = mean_changes[-stop_patience:]
        #         if all(abs(recent_mean_changes[i] - recent_mean_changes[i-1]) < early_stopping_threshold for i in range(1, len(recent_mean_changes))):
        #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
        #             break
        # else:
        #     mean_changes.append(mean_val)

    # Save results
    Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()

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
    
    return x_k , final_metric



def acce_RED_earlystop(   ##  best performence
    model, sampler, measurement_cond_fn, ref_img, y_n , device, model_config,measure_config,operator,fname,
    iter_step=4, iteration=1000, denoiser_step=10, stop_patience=40, 
    early_stopping_threshold=0.0, lr=0.02, out_path='./outputs/', mask=None, random_seed=None):

    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # 创建早停实例
    early_stopping = EarlyStopping(patience=stop_patience, min_delta=early_stopping_threshold, verbose=True)
    
    # 初始化
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    losses = []

    # alpha = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device,requires_grad=True)


    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    mu = torch.tensor(0.5, requires_grad=True, device=device)

    # optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.Adam([Z], lr=0.02)

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
    # optimizer = torch.optim.Adam([sample], lr=lr)

    optimizer = torch.optim.Adam([{'params': sample, 'lr': lr}, {'params': alpha, 'lr': lr}])    
    
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
        
        difference = y_n - operator.forward(x_t)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
            
        v_k = x_t - mu*  norm_grad
        
        x_k = alpha * x_t + (1-alpha) * v_k
        
        val_loss = 0.0
        
        # 计算损失并优化
        if mask is not None:
        # loss = criterion(operator.forward(x_t), y_n)
            loss = l1_loss(operator.forward(x_k, mask=mask), y_n)
        else:            
            loss = l1_loss(operator.forward(x_k), y_n)
            
        val_loss = loss.item()
        
        # loss.backward()
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

         # 调用早停策略
        early_stopping(val_loss)

        if early_stopping.stop_training():
            print(f"Training stopped at epoch {epoch+1} due to early stopping.")
            break
        
        # # 早停机制
        # mean_val = np.mean(output_numpy)
        # if epoch > 0:
        #     mean_changes.append(mean_val)
        #     if len(mean_changes) >= stop_patience:
        #         recent_mean_changes = mean_changes[-stop_patience:]
        #         if all(abs(recent_mean_changes[i] - recent_mean_changes[i-1]) < early_stopping_threshold for i in range(1, len(recent_mean_changes))):
        #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
        #             break
        # else:
        #     mean_changes.append(mean_val)

    # Save results
    Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()
    
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    
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
    
    return x_k , final_metric


# def acce_RED_diff_pro(   #####   effective with hard param mu
#     model, sampler, measurement_cond_fn, ref_img, y_n , device, model_config,measure_config,operator,fname,
#     iter_step=4, iteration=1000, denoiser_step=10, stop_patience=5, 
#     early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None,random_seed=None):

    
    
#     # 使用传入的随机种子重新设置随机种子
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed)
    
#     # 初始化
#     Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device)
#     criterion = torch.nn.MSELoss().to(device)
#     l1_loss = torch.nn.L1Loss()
#     losses = []

#     # alpha = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device,requires_grad=True)


#     alpha = torch.tensor(0.5, requires_grad=True, device=device)
#     mu = torch.tensor(0.5, requires_grad=True, device=device)
#     gamma = torch.tensor(0.1, requires_grad=True, device=device)
#     # optimizer = torch.optim.Adam([x], lr=0.01)
#     # optimizer = torch.optim.Adam([Z], lr=0.02)

#     # 初始采样过程
#     with torch.no_grad():  # 禁用梯度计算
#         for i, t in enumerate(tqdm(list(range(denoiser_step))[::-1], desc="Denoising Steps")):
#             time = torch.tensor([t] * ref_img.shape[0], device=device)
#             if i >= denoiser_step - iter_step:
#                 print(f"Stopping at the last {iter_step} steps (step {i+1})")
#                 break
#             if i == 0:
#                 sample, pred_start = sampler.p_sample(
#                     model=model, x=Z, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
#                 )
#             else:
#                 sample, pred_start = sampler.p_sample(
#                     model=model, x=sample, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
#                 )
                
#             metrics = compute_metrics(
#                 sample=sample,
#                 ref_img=ref_img,
#                 out_path=out_path,
#                 device=device,
#                 loss_fn_alex=loss_fn_alex,
#                 metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
#             )

#     # 优化过程
#     sample = sample.detach().clone().requires_grad_(True)
#     # optimizer = torch.optim.Adam([sample], lr=lr)
#     b_k_list = [y_n]

#     optimizer = torch.optim.Adam([{'params': sample, 'lr': lr}, {'params': alpha, 'lr': lr},{'params': gamma, 'lr': lr}])    

#     # 定义 ReduceLROnPlateau 调度器
#     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)
    
#     for epoch in tqdm(range(iteration), desc="Training Epochs"):
#         model.eval()
#         x_t = sample
#         x_t.requires_grad_(True)
#         optimizer.zero_grad()

        
#         # if epoch == 0:
#         #     b_t_pre = x_t
#         # else:
#         #     b_t_pre = b_t
            
#         x_t_pre = x_t
#         # 更新 x_t
#         for i, t in enumerate(list(range(iter_step))[::-1]):
#             time = torch.tensor([t] * ref_img.shape[0], device=device)
#             x_t, pred_start = sampler.p_sample(
#                 model=model, x=x_t, t=time, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
#             )
        
#         difference = y_n - operator.forward(x_t)
#         norm = torch.linalg.norm(difference)
#         norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
            
#         b_t = x_t - mu*  norm_grad
        

#         p_t = b_t + gamma*(b_t - x_t)
        
#         # p_t = b_t + gamma*(b_t - b_t_pre)
        
#         x_k = alpha*x_t + (1-alpha) * p_t
        
#         # 计算损失并优化
#         if mask is not None:
#         # loss = criterion(operator.forward(x_t), y_n)
#             loss = l1_loss(operator.forward(x_k, mask=mask), y_n)
#         else:            
#             loss = l1_loss(operator.forward(x_k), y_n)

#         # loss.backward()
#         loss.backward(retain_graph=True)

#         optimizer.step()
#         losses.append(loss.item())

#         # 计算并记录指标
#         with torch.no_grad():
#             metrics = compute_metrics(
#                 sample=x_k,
#                 ref_img=ref_img,
#                 out_path=out_path,
#                 device=device,
#                 loss_fn_alex=loss_fn_alex,
#                 metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
#             )

#         # # 早停机制
#         # mean_val = np.mean(output_numpy)
#         # if epoch > 0:
#         #     mean_changes.append(mean_val)
#         #     if len(mean_changes) >= stop_patience:
#         #         recent_mean_changes = mean_changes[-stop_patience:]
#         #         if all(abs(recent_mean_changes[i] - recent_mean_changes[i-1]) < early_stopping_threshold for i in range(1, len(recent_mean_changes))):
#         #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
#         #             break
#         # else:
#         #     mean_changes.append(mean_val)

#     # Save results
#     Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
#     plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
#     plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

#     # # Plot losses
#     # plt.plot(losses, label='Loss')
#     # plt.legend()
#     # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

#     # # Plot PSNR values
#     # plt.plot(metrics['psnr'])
#     # plt.savefig(os.path.join(out_path, 'psnr.png'))
#     # plt.close()
    
#     plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
#     plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
#     # 转换最佳图像和参考图像为 numpy 格式
#     best_img_np = x_t.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
#     ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

#     # 计算 PSNR
#     final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
#     # 计算 SSIM
#     final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
#     # 计算 LPIPS
#     best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#     ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
#     final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

#     # 将结果组织到字典中
#     final_metric = {
#         'psnr': final_psnr,
#         'ssim': final_ssim,
#         'lpips': final_lpips
#     }
    
    
#     # 打印最终的 PSNR, SSIM, LPIPS
#     print(f"Final metrics between best reconstructed image and reference image:")
#     print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
#     return x_k , final_metric




def acce_DMPlug(
    model, sampler, measurement_cond_fn, ref_img, y_n , device, model_config,measure_config,operator,fname,
    iter_step=4, iteration=1000, denoiser_step=10, stop_patience=5, 
    early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None):

    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # 初始化
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    criterion = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss()
    losses = []

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
    optimizer = torch.optim.Adam([sample], lr=lr)

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
        
        # 计算损失并优化
        if mask is not None:
        # loss = criterion(operator.forward(x_t), y_n)
            loss = l1_loss(operator.forward(x_t, mask=mask), y_n)
        else:            
            loss = l1_loss(operator.forward(x_t), y_n)

        # loss.backward()
        loss.backward(retain_graph=True)

        optimizer.step()
        losses.append(loss.item())

        # 计算并记录指标
        with torch.no_grad():
            metrics = compute_metrics(
                sample=x_t,
                ref_img=ref_img,
                out_path=out_path,
                device=device,
                loss_fn_alex=loss_fn_alex,
                metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
            )

        # # 早停机制
        # mean_val = np.mean(output_numpy)
        # if epoch > 0:
        #     mean_changes.append(mean_val)
        #     if len(mean_changes) >= stop_patience:
        #         recent_mean_changes = mean_changes[-stop_patience:]
        #         if all(abs(recent_mean_changes[i] - recent_mean_changes[i-1]) < early_stopping_threshold for i in range(1, len(recent_mean_changes))):
        #             print(f"Early stopping triggered after {epoch + 1} epochs due to mean stability.")
        #             break
        # else:
        #     mean_changes.append(mean_val)

    # Save results
    # Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    # plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_t))

    # # Plot losses
    # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.savefig(os.path.join(out_path, f"loss_{fname.split('.')[0]}.png"))    

    # # Plot PSNR values
    # plt.plot(metrics['psnr'])
    # plt.savefig(os.path.join(out_path, 'psnr.png'))
    # plt.close()
    
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
    
    
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
    
    return x_t , final_metric






def mpgd(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None, writer=None, img_index=None):
    """
    采样、计算评价指标并保存结果
    
    Parameters:
    - sample_fn: 采样函数
    - ref_img: 参考图像张量
    - y_n: 噪声后的图像张量
    - out_path: 输出保存路径
    - fname: 保存的文件名
    - device: 运行的设备（CPU 或 GPU）
    - mask: 可选的掩码张量
    - random_seed: 随机种子
    - writer: TensorBoard SummaryWriter对象
    - img_index: 当前处理的图像索引，用于TensorBoard日志
    """
    
    # 使用传入的随机种子重新设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 开始采样
    x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
    sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, 
                      mask=mask, ref_img=ref_img, writer=writer, img_index=img_index)

    # 初始化评价指标列表
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        best_img_np = sample.cpu().squeeze().detach().numpy().transpose(1, 2, 0) 
        ref_img_np = ref_img.cpu().squeeze().numpy().transpose(1, 2, 0)

        final_psnr = peak_signal_noise_ratio(ref_img_np, best_img_np)
        final_ssim = structural_similarity(ref_img_np, best_img_np, channel_axis=2, data_range=1)
        best_img_torch = torch.from_numpy(best_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        ref_img_torch = torch.from_numpy(ref_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        final_lpips = loss_fn_alex(ref_img_torch, best_img_torch).item()

        # 记录最终指标到 TensorBoard
        metrics = {
            'psnr': final_psnr,
            'ssim': final_ssim,
            'lpips': final_lpips
        }
        
        if writer is not None:
            # 使用图像索引作为标识符记录最终指标
            writer.add_scalar(f'Final/PSNR', final_psnr, img_index)
            writer.add_scalar(f'Final/SSIM', final_ssim, img_index)
            writer.add_scalar(f'Final/LPIPS', final_lpips, img_index)
            
            # 添加图像到TensorBoard - 每种类型放在单独的文件夹
            writer.add_image(f'Reference/image_{img_index}', ref_img_torch[0], 0)
            writer.add_image(f'Reconstructed/image_{img_index}', best_img_torch[0], 0)
            writer.add_image(f'Noisy/image_{img_index}', y_n[0], 0)
    
    # 保存结果和图像
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))

    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, metrics



def DPS(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None, writer=None, img_index=None):
    """
    采样、计算评价指标并保存结果
    
    Parameters:
    - sample_fn: 采样函数
    - ref_img: 参考图像张量
    - y_n: 噪声后的图像张量
    - out_path: 输出保存路径
    - fname: 保存的文件名
    - device: 运行的设备（CPU 或 GPU）
    - mask: 可选的掩码
    - random_seed: 随机种子
    - writer: TensorBoard SummaryWriter对象
    - img_index: 当前处理的图像索引，用于TensorBoard日志
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
        writer.add_scalar(f'Final/PSNR', final_psnr, img_index)
        writer.add_scalar(f'Final/SSIM', final_ssim, img_index)
        writer.add_scalar(f'Final/LPIPS', final_lpips, img_index)
        
        # 添加最终重建图像到TensorBoard
        writer.add_image(f'Images/Reconstructed_{img_index}', best_img_torch[0], img_index)
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, final_metric



def compute_metrics(sample, ref_img, out_path, device, loss_fn_alex, epoch,iteration, metrics=None):
    """
    初始化指标列表和最佳图像变量，计算 PSNR、SSIM 和 LPIPS 指标，保存结果到 CSV 文件，并跟踪最佳图像。
    
    参数:
        sample (torch.Tensor): 当前的采样结果张量。
        ref_numpy (np.array): 参考图像的 numpy 数组表示。
        out_path (str): 指标 CSV 文件保存路径。
        device (torch.device): 设备，通常为 'cuda' 或 'cpu'。
        loss_fn_alex (lpips.LPIPS): LPIPS 计算的损失函数。
        metrics (dict, optional): 包含指标列表和最佳图像信息的字典，若为 None，则自动初始化。
    
    返回:
        dict: 更新后的包含 PSNR、SSIM、LPIPS 列表
    """
    # 初始化指标列表和最佳图像变量（如果未提供）
    if metrics is None:
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

    # 转换采样结果为 numpy 格式
    ref_numpy = ref_img.cpu().squeeze().numpy()
    output_numpy = sample.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    ref_numpy_cal = ref_numpy.transpose(1, 2, 0)

    # 计算 PSNR
    tmp_psnr = peak_signal_noise_ratio(ref_numpy_cal, output_numpy)
    metrics['psnr'].append(tmp_psnr)

    # 计算 SSIM
    tmp_ssim = structural_similarity(ref_numpy_cal, output_numpy, channel_axis=2, data_range=1)
    metrics['ssim'].append(tmp_ssim)

    # 计算 LPIPS
    rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    gt_img_torch = torch.from_numpy(ref_numpy_cal).permute(2, 0, 1).unsqueeze(0).float().to(device)
    lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
    metrics['lpips'].append(lpips_alex)

    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)
    
    # 将结果写入 CSV 文件
    file_path = os.path.join(out_path, "metrics_curve.csv")
    # 如果 iteration 达到指定值，则清空文件内容
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        # 在文件为空时添加标题行
        if os.path.getsize(file_path) == 0:
            writer.writerow(["PSNR", "SSIM", "LPIPS"])
        writer.writerow([tmp_psnr, tmp_ssim, lpips_alex])
    if epoch == iteration-1:  # 替换 YOUR_THRESHOLD 为你希望的 iteration 阈值
        open(file_path, 'w').close()  # 清空文件
    return metrics
    

class EarlyStopping:
    def __init__(self, patience=40, min_delta=0, verbose=False):
        """
        初始化早停策略
        
        参数：
        - patience: 等待多少个 epoch 后，如果验证指标不再改善，则停止训练。
        - min_delta: 允许的最小变化量，低于该变化量时不认为是改进。
        - verbose: 是否打印早停信息。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """
        调用时，检查当前的指标是否足够好来更新最优指标。
        """
        if self.best_score is None:
            self.best_score = metric
        elif metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0  # reset counter if metric improves
        else:
            self.counter += 1  # increase counter if no improvement

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")

    def stop_training(self):
        return self.early_stop
    