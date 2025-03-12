import torch
import os
import csv
import time  
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
import torch.nn as nn 


# 移动方差早停 (ES-WMV: Early Stopping with Weighted Moving Variance)
class ESWithWMV:
    """
    基于加权移动方差的早停策略
    
    参数:
        window_size (int): 用于计算移动方差的窗口大小
        var_threshold (float): 方差早停的阈值
        alpha (float): 损失早停的相对改进阈值
        patience (int): 允许没有足够改进的迭代次数
        min_epochs (int): 在考虑早停前的最小迭代次数
        verbose (bool): 是否打印详细信息
    """
    def __init__(self, window_size=15, var_threshold=0.0005, alpha=0.005, 
                 patience=20, min_epochs=50, verbose=True):
        self.window_size = window_size
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        
        # 图像历史
        self.image_history = []
        
        # 损失历史和相关变量
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.stop_flag = False
        
    def calculate_weight(self, idx, window_size):
        """计算加权移动方差中的权重，越近的样本权重越大"""
        return (idx + 1) / sum(range(1, window_size + 1))
    
    def calculate_wmv(self):
        """计算加权移动方差，更保守的实现"""
        if len(self.image_history) < self.window_size:
            return float('inf')
        
        # 获取最近的窗口大小的图像
        recent_images = self.image_history[-self.window_size:]
        
        # 计算均值图像
        mean_image = sum(recent_images) / len(recent_images)
        
        # 计算加权方差（更保守的实现）
        weighted_var = 0
        total_pixels = 0
        
        # 计算每个像素位置的加权方差
        for i, img in enumerate(recent_images):
            weight = self.calculate_weight(i, self.window_size)
            diff = img - mean_image
            # 使用更保守的方差计算，考虑整个图像的局部变化
            weighted_var += weight * torch.mean(diff * diff).item()
            
            # 额外检查局部差异的最大值
            max_local_diff = torch.max(torch.abs(diff)).item()
            if max_local_diff > weighted_var:
                weighted_var = max(weighted_var, max_local_diff * 0.1)
            
            total_pixels += 1
        
        # 如果还没有足够多的样本，返回一个更大的值
        if len(self.image_history) < self.window_size * 2:
            return weighted_var * 2.0
            
        return weighted_var
    
    def __call__(self, epoch, image, loss):
        """
        检查是否应该早停 - 更保守的实现
        
        参数:
            epoch (int): 当前迭代轮次
            image (torch.Tensor): 当前图像
            loss (float): 当前损失值
            
        返回:
            bool: 如果应该停止则返回True，否则返回False
        """
        # 存储当前图像和损失
        self.image_history.append(image.detach().clone())
        self.loss_history.append(loss)
        
        # 如果迭代次数少于最小迭代次数，继续训练
        if epoch < self.min_epochs:
            return False
        
        # 计算加权移动方差
        current_wmv = self.calculate_wmv()
        
        # 记录WMV但不总是触发早停
        if current_wmv < self.var_threshold and epoch > self.min_epochs * 1.5:
            if self.verbose:
                print(f"WMV指标低于阈值: {current_wmv:.6f}，但继续训练")
            
            # 只有当方差连续多轮都非常低，并且已经训练了很多轮时才触发
            if current_wmv < self.var_threshold * 0.5 and len(self.loss_history) > 5:
                # 检查最近5轮的损失变化是否也很小
                recent_losses = self.loss_history[-5:]
                loss_variance = np.var(recent_losses)
                
                if loss_variance < self.alpha * 0.01 and epoch > self.min_epochs * 2:
                    if self.verbose:
                        print(f"WMV和损失方差都很低，早停触发。WMV={current_wmv:.6f}, 损失方差={loss_variance:.6f}")
                    self.stop_flag = True
                    return True
        
        # 更新最佳损失 - 更保守的更新
        if loss < self.best_loss * (1 - self.alpha):
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 在训练前期，计数器增加得更慢
            if epoch < self.min_epochs * 1.5:
                self.counter += 0.5  # 半速计数
            else:
                self.counter += 1
            
        # 检查损失耐心早停条件 - 确保连续多轮无改进
        if self.counter >= self.patience:
            # 额外检查最近N轮的损失变化率
            if len(self.loss_history) >= self.patience:
                recent_losses = self.loss_history[-self.patience:]
                avg_change_rate = np.mean([abs(recent_losses[i] - recent_losses[i-1])/recent_losses[i-1] 
                                        for i in range(1, len(recent_losses))])
                
                # 如果损失还在显著变化，给予额外的耐心
                if avg_change_rate > self.alpha * 2:
                    if self.verbose:
                        print(f"损失仍在变化({avg_change_rate:.6f})，重置耐心计数器")
                    self.counter = self.patience // 2
                    return False
            
            if self.verbose:
                print(f"损失早停触发，{self.patience}轮没有足够改进")
            self.stop_flag = True
            return True
            
        return False
    
    def get_best_epoch(self):
        """返回具有最佳损失的轮次"""
        return self.best_epoch
    
    def should_stop(self):
        """返回是否应该停止训练"""
        return self.stop_flag
    
    def reset(self):
        """重置早停器"""
        self.image_history = []
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.stop_flag = False

def log_metrics_to_tensorboard(writer, metrics, step, img_index=None, prefix=''):
    """将指标记录到TensorBoard
    
    参数:
        writer: TensorBoard SummaryWriter对象
        metrics: 包含指标值的字典
        step: 训练步骤
        img_index: 可选的图像索引，用于区分不同图像
        prefix: 指标名称前缀
    """
    if writer is None:
        return  # 如果没有提供writer，直接返回
        
    # 为每个指标添加适当的前缀（如果提供）
    tag_prefix = f"{prefix}/" if prefix else ""
    
    # 如果提供了图像索引，在标签中加入图像索引
    if img_index is not None:
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                # 如果值是列表，取最后一个值
                value = value[-1] if value else 0
            writer.add_scalar(f"{tag_prefix}{metric_name}/image_{img_index}", value, step)
    else:
        # 没有图像索引，直接记录指标
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                # 如果值是列表，取最后一个值
                value = value[-1] if value else 0
            writer.add_scalar(f"{tag_prefix}{metric_name}", value, step)

def plot_and_log_curves(writer, losses, psnrs, ssims, lpipss, out_path, img_index=None):
    """绘制并记录训练曲线到TensorBoard
    
    参数:
        writer: TensorBoard SummaryWriter对象
        losses: 损失值列表
        psnrs: PSNR值列表
        ssims: SSIM值列表
        lpipss: LPIPS值列表
        out_path: 输出路径
        img_index: 可选的图像索引
    """
    # 确保路径存在
    import os
    os.makedirs(out_path, exist_ok=True)
    
    # 避免matplotlib导入问题
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # 绘制训练曲线
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
    
    # 特定于图像的曲线保存路径
    if img_index is not None:
        curves_path = os.path.join(out_path, f'curves_image_{img_index}.png')
    else:
        curves_path = os.path.join(out_path, 'curves.png')
    
    plt.savefig(curves_path)
    plt.close()
    
    # 将曲线图添加到TensorBoard
    if writer is not None:
        try:
            curves_img = plt.imread(curves_path)
            img_tensor = torch.from_numpy(curves_img)
            
            # 确保图像具有正确的维度
            if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:  # [H, W, C]
                img_tensor = img_tensor.permute(2, 0, 1)  # 转换为[C, H, W]
            elif img_tensor.dim() == 2:  # [H, W]
                img_tensor = img_tensor.unsqueeze(0)  # 转换为[1, H, W]
                
            if img_index is not None:
                writer.add_image(f'Curves/image_{img_index}', img_tensor, 0)
            else:
                writer.add_image('Curves', img_tensor, 0)
        except Exception as e:
            print(f"向TensorBoard添加曲线图像时出错: {e}")
        
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

    # 确保样本和参考图像具有正确的维度
    # 首先检查并标准化样本图像
    if sample.dim() == 3:  # 如果样本是[C, H, W]
        sample = sample.unsqueeze(0)  # 转换为[1, C, H, W]
    
    # 确保ref_img也是[B, C, H, W]格式
    if ref_img.dim() == 3:  # 如果参考图像是[C, H, W]
        ref_img = ref_img.unsqueeze(0)  # 转换为[1, C, H, W]
    
    # 转换为numpy格式计算指标
    ref_numpy = ref_img.cpu().squeeze().numpy()
    
    # 确保样本图像有正确的形状用于numpy处理
    output_numpy = sample.detach().cpu().squeeze().numpy()
    
    # 检查输出是否仍然是[C, H, W]格式
    if output_numpy.ndim == 3 and output_numpy.shape[0] == 3:
        output_numpy = output_numpy.transpose(1, 2, 0)  # 转换为[H, W, C]用于计算指标
    
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
        print(f"ref_numpy_cal形状: {ref_numpy_cal.shape}")
        print(f"output_numpy形状: {output_numpy.shape}")
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

def save_image(img_tensor, path, normalize=True):
    """
    保存张量图像到文件系统，处理各种格式和维度问题
    
    参数:
        img_tensor (torch.Tensor): 图像张量，可以是[B, C, H, W], [C, H, W]或[H, W, C]格式
        path (str): 保存路径
        normalize (bool): 是否规范化图像到[0, 1]范围
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 确保路径存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # 将tensor转换到CPU并分离梯度
    img = img_tensor.detach().cpu()
    
    # 确保图像有正确的维度
    if img.dim() == 4:  # [B, C, H, W]
        img = img.squeeze(0)  # 转换为[C, H, W]
    
    # 转换为numpy数组
    img_np = img.numpy()
    
    # 检查并调整图像格式
    if img_np.shape[0] == 3 and len(img_np.shape) == 3:  # [C, H, W]
        img_np = np.transpose(img_np, (1, 2, 0))  # 转换为[H, W, C]
    
    # 规范化图像
    if normalize:
        img_np = normalize_image(img_np)
    
    # 确保图像值在有效范围内
    img_np = np.clip(img_np, 0, 1)
    
    # 保存图像
    try:
        # 使用PIL保存
        if img_np.shape[-1] == 3:  # RGB图像
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(path)
        else:  # 灰度图像
            plt.imsave(path, img_np, cmap='gray')
    except Exception as e:
        print(f"保存图像到{path}时出错: {e}")
        # 尝试使用matplotlib作为备选
        plt.figure(figsize=(10, 10))
        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            plt.imshow(img_np)
        else:
            plt.imshow(img_np, cmap='gray')
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

def normalize_image(img):
    """
    将图像规范化到[0, 1]范围
    
    参数:
        img: numpy数组，图像数据
    
    返回:
        规范化后的图像
    """
    img_min = img.min()
    img_max = img.max()
    
    # 避免除以零
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img

def clear_color_fixed(x):
    """
    处理tensor图像用于显示，解决维度问题
    
    参数:
        x (torch.Tensor): 输入图像张量
        
    返回:
        numpy.ndarray: 处理后的图像，格式为[H, W, C]，范围在[0, 1]
    """
    import torch
    import numpy as np
    
    # 处理复数tensor
    if torch.is_complex(x):
        x = torch.abs(x)
    
    # 分离梯度并转移到CPU
    x = x.detach().cpu()
    
    # 处理batch维度
    if x.dim() == 4:
        x = x.squeeze(0)
    
    # 转换为numpy
    x_np = x.numpy()
    
    # 调整通道顺序
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:  # [C, H, W]
        x_np = np.transpose(x_np, (1, 2, 0))
        
    # 如果是单通道，去掉通道维度
    if x_np.shape[-1] == 1:
        x_np = x_np.squeeze(-1)
    
    # 规范化并返回
    return normalize_image(x_np)

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
    # 初始化早停策略

    stop_patience = 10
    early_stopping_threshold = 0.01
    early_stopper = EarlyStopping(patience=stop_patience, min_delta=early_stopping_threshold, verbose=True)
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
        
        # 记录参考图像和测量图像不需要修改，因为这些是算法的输入
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
                
                # 每隔一定轮数记录中间过程图像
                if epoch % 100 == 0 or epoch == iteration - 1:
                    writer.add_image(f'DMPlug/Image_{img_index}/Intermediate/Epoch_{epoch}', 
                                   (sample[0] + 1)/2, epoch)
                
                # 只记录最佳PSNR的中间结果，不记录最终的重建图像（留给主函数记录）
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
    
    # 记录训练曲线，这部分保留，因为它记录的是算法内部的过程数据
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
        writer.add_image(f'DMPlug/Image_{img_index}/Learning_Curves', 
                       torch.from_numpy(curves_img).permute(2, 0, 1), 0)
        
        # 删除记录误差图的部分
        
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
        window_size=8,                         # 使用更大窗口(20)计算加权移动方差，提高稳定性
        var_threshold=0.002,  # 降低方差阈值，使其更难触发
        alpha=0.01,     # 降低相对改进阈值，使其更容易满足条件
        patience=8,               # 增加耐心参数，允许更长时间无改进
        min_epochs=30,                         # 显著增加最小训练轮数，确保充分训练
        verbose=True                           # 打印详细信息
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
                
                # 检查是否应该早停 - 添加更保守的条件
                if early_stopper(epoch, x_k, loss.item()):
                    # 额外检查：如果训练不足200轮且损失还在显著下降，继续训练
                    if epoch < 30 and len(losses) > 5:  # 减少到30轮和5个样本检查
                        recent_losses = losses[-5:]
                        loss_trend = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
                        
                        if loss_trend > 0.1:  # 提高阈值到10%，确保显著下降才继续
                            print(f"早停被触发，但损失仍在显著下降({loss_trend:.2%})，继续训练")
                            # 部分重置早停器以继续训练，但不完全重置
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
        


    # Save results
    Z_np = (Z.detach().cpu().numpy().squeeze(0)).clip(0, 1)
    plt.imsave(os.path.join(out_path, 'Z_image.png'), Z_np.transpose(1, 2, 0))
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(x_k))

    
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


# 修改mpgd函数，移除冗余的重建图像和误差图像记录
def mpgd(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None, writer=None, img_index=None):
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
            from util.img_utils import clear_color
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


# 修改DPS函数，移除冗余记录
def DPS(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None, writer=None, img_index=None):
    """
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
    
    # 将最终指标记录到TensorBoard，但移除最终重建图像的记录
    if writer is not None and img_index is not None:
        writer.add_scalar(f'DPS/Final/PSNR', final_psnr, img_index)
        writer.add_scalar(f'DPS/Final/SSIM', final_ssim, img_index)
        writer.add_scalar(f'DPS/Final/LPIPS', final_lpips, img_index)
    
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, final_metric

    