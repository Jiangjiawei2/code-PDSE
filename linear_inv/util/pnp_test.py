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
    Z = torch.randn((1, 3, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}])
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
        
        # Loss calculation
        if mask is not None:
        # loss = criterion(operator.forward(x_t), y_n)
            loss = criterion(operator.forward(sample, mask=mask), y_n)
        else:            
            loss = criterion(operator.forward(sample), y_n)

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
            epoch=epoch,
            iteration=iteration,
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
                
            # metrics = compute_metrics(
            #     sample=sample,
            #     ref_img=ref_img,
            #     out_path=out_path,
            #     device=device,
            #     loss_fn_alex=loss_fn_alex,
            #     metrics=None  # 初次调用时不传递 metrics，函数会自动初始化
            # )

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
            sample=sample,
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






def mpgd(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None):
    """
    采样、计算评价指标并保存结果
    
    Parameters:
    - sample_fn: 采样函数
    - ref_img: 参考图像张量
    - y_n: 噪声后的图像张量
    - out_path: 输出保存路径
    - fname: 保存的文件名
    - device: 运行的设备（CPU 或 GPU）
    """
    
    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # 开始采样
    x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
    sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path,mask=mask)

    # 初始化评价指标列表
    psnrs = []
    ssims = []
    lpipss = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # 计算并记录指标

    # Save results
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

    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
        
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    
    
    return sample , final_metric



def DPS(sample_fn, ref_img, y_n, out_path, fname, device, mask=None, random_seed=None):
    """
    采样、计算评价指标并保存结果
    
    Parameters:
    - sample_fn: 采样函数
    - ref_img: 参考图像张量
    - y_n: 噪声后的图像张量
    - out_path: 输出保存路径
    - fname: 保存的文件名
    - device: 运行的设备（CPU 或 GPU）
    """
    
    # 使用传入的随机种子重新设置随机种子
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # 开始采样
    x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
    sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path,mask=mask)

    # 初始化评价指标列表
    psnrs = []
    ssims = []
    lpipss = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # 计算并记录指标

    # Save results
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

    final_metric = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
        
    # 打印最终的 PSNR, SSIM, LPIPS
    print(f"Final metrics between best reconstructed image and reference image:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    
    
    return sample , final_metric



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
    