import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import lpips
from util.img_utils import clear_color
from .posterior_mean_variance import get_mean_processor, get_var_processor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .gaussian_diffusion import extract_and_expand

__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing="",
                   eta=1.0):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps,
                   eta=eta)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 eta=1.0,
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.eta = eta
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = img.requires_grad_()
            #unconditional sampling
            out = self.p_sample(x=img, t=time, model=model) #out = {x_t-1, x_0_hat}
            
            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)
            
            alpha_bar = self.alphas_cumprod[time]
            
            if idx < 0.1*self.num_timesteps:
                small_t = True
            else:
                small_t = False

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'],
                                      at=alpha_bar,
                                      small_t=small_t)
            img = img.detach_()
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img       
        
    def p_sample(self, model, x, t):
        raise NotImplementedError
    
    # def predict_eps_from_x_start(self, x_t, t, pred_xstart):
    #     raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}
    

@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        eta = self.eta
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        
        
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2
    
@register_sampler(name='x0ddim')
class DDIMx0(SpacedDiffusion):
    # @torch.no_grad()
    def p_sample(self, model, x, t, measurement,
                      measurement_cond_fn, eta=1.0, mask = None,):
        eta = self.eta
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # # # #############  grad  #################
        # x_0_hat = out['pred_xstart']
        
        # with torch.enable_grad():
        #     x_0_hat = x_0_hat.requires_grad_()
        #     x0_t, distance = measurement_cond_fn(measurement=measurement,
        #                                 x_0_hat=x_0_hat,
        #                                 at=alpha_bar_prev,
        #                                 t=t/self.num_timesteps)
        
        # out["pred_xstart"] = x0_t
        
        #########################################

        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        # if mask is not None:
        #     ori_in = measurement
        #     img_orig = self.q_sample(ori_in , t)
        #     # alpha_bar_prev.sqrt() * ori_in + torch.sqrt(1 - alpha_bar_prev) * eps
        #     sample = img_orig * mask + (1. - mask) * sample

           
        return  sample,  out["pred_xstart"]
        # return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    @torch.no_grad()
    def p_sample_loop(self,
                    model,
                    x_start,
                    measurement,
                    measurement_cond_fn,
                    record,
                    save_root,
                    mask=None,
                    ref_img=None,
                    writer=None,
                    img_index=None,
                    pbar=None):
        """
        增强版采样循环，记录中间过程的损失和重建图像
        
        Parameters:
        - model: 扩散模型
        - x_start: 初始噪声图像
        - measurement: 测量数据
        - measurement_cond_fn: 引导函数
        - record: 是否记录中间结果
        - save_root: 保存路径
        - mask: 可选掩码
        - ref_img: 参考图像，用于计算指标
        - writer: TensorBoard SummaryWriter对象
        - img_index: 当前处理的图像索引
        """
        # 导入所需的库
        import torch
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import lpips
        from tqdm import tqdm
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        eta = self.eta
        img = x_start
        device = x_start.device
        model.eval()
        
        # 转换图像索引为整数，默认为0
        if img_index is None:
            img_index = 0
        else:
            img_index = int(img_index)
        
        # 用于存储中间过程的损失和图像
        intermediate_losses = []
        intermediate_psnrs = []
        intermediate_ssims = []
        intermediate_lpips = []
        
        # 如果使用TensorBoard，记录初始配置
        if writer is not None:
            writer.add_text(f'DDIMx0/Image_{img_index}/Config', 
                        f'Eta: {eta}\n'
                        f'Mask: {"Yes" if mask is not None else "No"}', 0)
            
            # 记录初始噪声图像
            writer.add_image(f'DDIMx0/Initial_Noise/Image_{img_index}', 
                            (img[0] + 1) / 2, 0)
            
            # 记录输入测量图像
            writer.add_image(f'DDIMx0/Measurement/Image_{img_index}', 
                            (measurement[0] + 1) / 2, 0)
            
            if ref_img is not None:
                writer.add_image(f'DDIMx0/Reference/Image_{img_index}', 
                                (ref_img[0] + 1) / 2, 0)
        
        # 初始化LPIPS模型
        loss_fn_alex = None
        if ref_img is not None:
            try:
                loss_fn_alex = lpips.LPIPS(net='alex').to(device)
            except Exception as e:
                print(f"LPIPS模型初始化失败: {e}")
                loss_fn_alex = None
        
        # 进度条
        if pbar is None:
            pbar = tqdm(list(range(self.num_timesteps))[::-1], 
                        desc=f'DDIM Sampling (Image {img_index})')
        else:
            # 如果传入了进度条，使用传入的
            pass
        
        for step_idx, idx in enumerate(pbar):
            time = torch.tensor([idx] * img.shape[0], device=device)
            t = time
            x = img
            
            # 标准扩散采样步骤
            out = self.p_mean_variance(model, x, t)
            
            eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
            alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
            alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            
            # 应用条件约束
            x_0_hat = out['pred_xstart'].detach()
            with torch.enable_grad():
                x_0_hat = x_0_hat.requires_grad_()
                x0_t, distance = measurement_cond_fn(measurement=measurement,
                                    x_0_hat=x_0_hat,
                                    at=alpha_bar_prev,
                                    t=t/self.num_timesteps)
            
            out["pred_xstart"] = x0_t
            
            # 记录损失
            intermediate_losses.append(distance.item())
            
            # 计算损失和指标
            if ref_img is not None and loss_fn_alex is not None:
                try:
                    # 获取当前预测的x0
                    current_img = out['pred_xstart']
                    
                    # 确保维度正确并转换为numpy数组
                    if current_img.dim() == 4:
                        current_img_np = current_img.cpu().squeeze().detach().numpy()
                    else:
                        current_img_np = current_img.cpu().detach().numpy()
                    
                    if current_img_np.shape[0] == 3:  # [C, H, W]
                        current_img_np = np.transpose(current_img_np, (1, 2, 0))
                    
                    if ref_img.dim() == 4:
                        ref_img_np = ref_img.cpu().squeeze().numpy()
                    else:
                        ref_img_np = ref_img.cpu().numpy()
                    
                    if ref_img_np.shape[0] == 3:  # [C, H, W]
                        ref_img_np = np.transpose(ref_img_np, (1, 2, 0))
                    
                    # 计算PSNR
                    try:
                        rec_psnr = peak_signal_noise_ratio(ref_img_np, current_img_np)
                        intermediate_psnrs.append(rec_psnr)
                    except Exception as e:
                        print(f"PSNR计算错误: {e}")
                        rec_psnr = 0
                        intermediate_psnrs.append(rec_psnr)
                    
                    # 计算SSIM
                    try:
                        rec_ssim = structural_similarity(
                            ref_img_np, 
                            current_img_np, 
                            channel_axis=2, 
                            data_range=1
                        )
                        intermediate_ssims.append(rec_ssim)
                    except Exception as e:
                        print(f"SSIM计算错误: {e}")
                        try:
                            # 尝试使用multichannel参数（旧版skimage）
                            rec_ssim = structural_similarity(
                                ref_img_np, 
                                current_img_np, 
                                multichannel=True, 
                                data_range=1
                            )
                            intermediate_ssims.append(rec_ssim)
                        except Exception as e2:
                            print(f"备选SSIM计算也失败: {e2}")
                            rec_ssim = 0
                            intermediate_ssims.append(rec_ssim)
                    
                    # 计算LPIPS
                    try:
                        # 准备用于LPIPS的张量
                        if current_img.dim() == 3:  # [C, H, W]
                            current_img_torch = current_img.unsqueeze(0).to(device)
                        else:  # [B, C, H, W]
                            current_img_torch = current_img.to(device)
                        
                        if ref_img.dim() == 3:  # [C, H, W]
                            ref_img_torch = ref_img.unsqueeze(0).to(device)
                        else:  # [B, C, H, W]
                            ref_img_torch = ref_img.to(device)
                        
                        rec_lpips = loss_fn_alex(current_img_torch, ref_img_torch).item()
                        intermediate_lpips.append(rec_lpips)
                    except Exception as e:
                        print(f"LPIPS计算错误: {e}")
                        import traceback
                        traceback.print_exc()
                        rec_lpips = 0
                        intermediate_lpips.append(rec_lpips)
                    
                    # 更新进度条显示
                    if pbar is not None:
                        pbar.set_postfix({
                            'PSNR': f"{rec_psnr:.2f}", 
                            'SSIM': f"{rec_ssim:.4f}", 
                            'LPIPS': f"{rec_lpips:.4f}"
                        })
                
                except Exception as e:
                    print(f"指标计算过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 记录损失和步骤信息到TensorBoard
            if writer is not None:
                writer.add_scalar(f'DDIMx0/Distance/Image_{img_index}', 
                    distance.item(), step_idx)
                
                # 如果有计算指标，也记录指标
                if len(intermediate_psnrs) > 0 and step_idx < len(intermediate_psnrs):
                    writer.add_scalar(f'DDIMx0/Metrics/PSNR/Image_{img_index}', 
                                    intermediate_psnrs[-1], step_idx)
                
                if len(intermediate_ssims) > 0 and step_idx < len(intermediate_ssims):
                    writer.add_scalar(f'DDIMx0/Metrics/SSIM/Image_{img_index}', 
                                    intermediate_ssims[-1], step_idx)
                
                if len(intermediate_lpips) > 0 and step_idx < len(intermediate_lpips):
                    writer.add_scalar(f'DDIMx0/Metrics/LPIPS/Image_{img_index}', 
                                    intermediate_lpips[-1], step_idx)
                
                # 记录中间重建图像（每隔一定步骤）
                if step_idx % 5 == 0:
                    writer.add_image(
                        f'DDIMx0/Intermediate/Image_{img_index}/Step_{step_idx}', 
                        (out['pred_xstart'][0] + 1) / 2, step_idx
                    )
            
            # 标准DDIM采样步骤
            noise = torch.randn_like(x)
            mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            
            img = mean_pred
            if t != 0:
                img += sigma * noise
            
            img = img.detach_()
        
        # 最终重建图像处理和记录
        if writer is not None:
            # 记录最终重建图像
            writer.add_image(f'DDIMx0/Final_Reconstruction/Image_{img_index}', 
                            (img[0] + 1) / 2, 0)
            
            # 记录最终指标
            if intermediate_psnrs:
                writer.add_scalar(f'DDIMx0/Final/PSNR/Image_{img_index}', intermediate_psnrs[-1], 0)
            if intermediate_ssims:
                writer.add_scalar(f'DDIMx0/Final/SSIM/Image_{img_index}', intermediate_ssims[-1], 0)
            if intermediate_lpips:
                writer.add_scalar(f'DDIMx0/Final/LPIPS/Image_{img_index}', intermediate_lpips[-1], 0)
        
        return img
    
    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

    # def p_sample_loop(self,
    #                   model,
    #                   x_start,
    #                   measurement,
    #                   measurement_cond_fn,
    #                   record,
    #                   save_root,
    #                   mask = None):
    #     """
    #     The function used for sampling from noise.
    #     """ 
    #     img = x_start
    #     device = x_start.device

    #     pbar = tqdm(list(range(self.num_timesteps))[::-1])
    #     for idx in pbar:
    #         time = torch.tensor([idx] * img.shape[0], device=device)
            
    #         img = img.requires_grad_()
    #         out = self.p_sample(x=img, t=time, model=model)
            
    #         # Give condition.
    #         noisy_measurement = self.q_sample(measurement, t=time)

    #         # TODO: how can we handle argument for different condition method?
    #         img, distance = measurement_cond_fn(x_t=out['sample'],
    #                                   measurement=measurement,
    #                                   noisy_measurement=noisy_measurement,
    #                                   x_prev=img,
    #                                   x_0_hat=out['pred_xstart'])
    #         img = img.detach_()
           
    #         pbar.set_postfix({'distance': distance.item()}, refresh=False)
    #         if record:
    #             if idx % 10 == 0:
    #                 file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
    #                 plt.imsave(file_path, clear_color(img))

    #     return img       
    
# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
