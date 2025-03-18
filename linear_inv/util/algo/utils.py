# util/algo/utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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

def plot_and_log_curves(writer, losses, psnrs, ssims, lpipss, out_path, img_index=None, algo_name="Algorithm"):
    """绘制并记录训练曲线到TensorBoard
    
    参数:
        writer: TensorBoard SummaryWriter对象
        losses: 损失值列表
        psnrs: PSNR值列表
        ssims: SSIM值列表
        lpipss: LPIPS值列表
        out_path: 输出路径
        img_index: 可选的图像索引
        algo_name: 算法名称，用于标记图像
    """
    # 确保路径存在
    os.makedirs(out_path, exist_ok=True)
    
    # 避免matplotlib导入问题
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    
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
        curves_path = os.path.join(out_path, f'{algo_name}_curves_image_{img_index}.png')
    else:
        curves_path = os.path.join(out_path, f'{algo_name}_curves.png')
    
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
                writer.add_image(f'{algo_name}/Curves/image_{img_index}', img_tensor, 0)
            else:
                writer.add_image(f'{algo_name}/Curves', img_tensor, 0)
        except Exception as e:
            print(f"向TensorBoard添加曲线图像时出错: {e}")

def compute_metrics(sample, ref_img, out_path, device, loss_fn_alex, epoch=None, iteration=None, metrics=None):
    """计算PSNR、SSIM和LPIPS指标
    
    参数:
        sample: 当前的采样结果张量
        ref_img: 参考图像张量
        out_path: 指标CSV文件保存路径
        device: 设备
        loss_fn_alex: LPIPS计算的损失函数
        epoch: 当前的epoch
        iteration: 总迭代次数
        metrics: 包含指标列表的字典，若为None，则自动初始化
    
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
        tmp_psnr = peak_signal_noise_ratio(ref_numpy_cal, output_numpy)
        metrics['psnr'].append(tmp_psnr)
    except Exception as e:
        print(f"PSNR计算错误: {e}")
        tmp_psnr = 0
        metrics['psnr'].append(tmp_psnr)

    # 计算SSIM
    try:
        tmp_ssim = structural_similarity(ref_numpy_cal, output_numpy, channel_axis=2, data_range=1)
        metrics['ssim'].append(tmp_ssim)
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        tmp_ssim = 0
        metrics['ssim'].append(tmp_ssim)

    # 计算LPIPS
    try:
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

# 早停相关类
class EarlyStopping:
    """早停策略"""
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.should_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        
    def stop_training(self):
        return self.should_stop

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