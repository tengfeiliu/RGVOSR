import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)
from dataloaders.degradation_meta import build_degradation_vector_from_meta, kernel_blur_strength, to_jsonable

cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

class RealESRGAN_degradation(object):
    def __init__(self, opt_name='params_realesrgan.yml', device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device #torch.device('cpu')
        optk = self.opt['kernel_info']       

        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # second degradation
        self.second_order_prob = self.opt['second_order_prob']
        self.first_order_prob = self.opt['first_order_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)
    
    def color_jitter_pt(self, img, brightness, contrast, saturation, hue, return_meta=False):
        fn_idx = torch.randperm(4)
        meta = {"applied": True, "order": [int(v) for v in fn_idx.cpu().tolist()]}
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                meta["brightness"] = brightness_factor
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                meta["contrast"] = contrast_factor
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                meta["saturation"] = saturation_factor
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                meta["hue"] = hue_factor
                img = adjust_hue(img, hue_factor)
        if return_meta:
            return img, meta
        return img

    def random_augment(self, img_gt):
        # random horizontal flip
        # img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)

        return img_gt

    def random_kernels(self, return_meta=False):
        meta = {}
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            kernel_meta = {"type": "sinc", "kernel_size": kernel_size, "omega_c": float(omega_c)}
        else:
            kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            kernel_meta = {"type": "mixed", "kernel_size": kernel_size}
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel_meta["strength"] = kernel_blur_strength(kernel)
        meta["kernel1"] = kernel_meta

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            kernel2_meta = {"type": "sinc", "kernel_size": kernel_size, "omega_c": float(omega_c)}
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
            kernel2_meta = {"type": "mixed", "kernel_size": kernel_size}

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel2_meta["strength"] = kernel_blur_strength(kernel2)
        meta["kernel2"] = kernel2_meta

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            meta["final_sinc"] = {
                "applied": True,
                "kernel_size": kernel_size,
                "omega_c": float(omega_c),
                "strength": kernel_blur_strength(sinc_kernel),
            }
        else:
            sinc_kernel = self.pulse_tensor
            meta["final_sinc"] = {"applied": False, "strength": 0.0}

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2) 

        if return_meta:
            return kernel, kernel2, sinc_kernel, meta
        return kernel, kernel2, sinc_kernel

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=False, return_meta=False):
        """
        img_gt: GPU Tensor of shape (B, C, H, W) in [0, 1].
        """
        img_gt = img_gt.to(self.device)
        if img_gt.dim() == 3:
            img_gt = img_gt.unsqueeze(0)

        batch_size = img_gt.size(0)
        ori_h, ori_w = img_gt.size()[2:4]

        if return_meta:
            kernel1, kernel2, sinc_kernel, meta = self.random_kernels(return_meta=True)
        else:
            kernel1, kernel2, sinc_kernel = self.random_kernels()
            meta = None
        kernel1, kernel2, sinc_kernel = kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        
        #scale_final = random.randint(4, 16)
        scale_final = self.opt['scale']
        out = img_gt
        if return_meta:
            meta.update(
                {
                    "stage": "none",
                    "scale_final": scale_final,
                    "resize_bak": bool(resize_bak),
                    "batch_size": int(batch_size),
                    "original_size": [int(ori_h), int(ori_w)],
                }
            )

        none_rand_second = random.random()
        none_rand_first = random.random()
        if none_rand_second > self.second_order_prob and none_rand_first > self.first_order_prob:
            mode = random.choice(['bicubic'])
            out = F.interpolate(out, size=(ori_h//scale_final, ori_w//scale_final), mode=mode)
            if return_meta:
                meta["stage"] = "resize_only"
                meta["stage1"] = {
                    "blur_applied": False,
                    "resize": {"mode": mode, "scale": 1.0 / scale_final, "target_size": [ori_h // scale_final, ori_w // scale_final]},
                    "noise": {"type": "none"},
                    "jpeg_quality": [],
                }

        first_rand_first = random.random()
        first_rand_second = random.random()
        if first_rand_first < self.first_order_prob and first_rand_second > self.second_order_prob:
            if return_meta:
                meta["stage"] = "first_order"
            # ----------------------- The first degradation process ----------------------- #
            # blur
            if kernel1.dim() == 2:
                kernel1 = kernel1.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif kernel1.dim() == 3:
                kernel1 = kernel1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif kernel1.dim() == 4 and kernel1.size(0) == 1: # [1, 1, k, k]
                kernel1 = kernel1.repeat(batch_size, 1, 1, 1)

            out = filter2D(out.contiguous(), kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            resize_mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=resize_mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                if return_meta:
                    sigma = float(np.random.uniform(*self.opt['noise_range']))
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=[sigma, sigma], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    noise_meta = {"type": "gaussian", "sigma": sigma, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                if return_meta:
                    scale_noise = float(np.random.uniform(*self.opt['poisson_scale_range']))
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=[scale_noise, scale_noise],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    noise_meta = {"type": "poisson", "scale": scale_noise, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            final_resize_mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=final_resize_mode)
            if return_meta:
                meta["stage1"] = {
                    "blur_applied": True,
                    "kernel": meta["kernel1"],
                    "resize": {"mode": resize_mode, "updown_type": updown_type, "scale": float(scale)},
                    "noise": noise_meta,
                    "jpeg_quality": to_jsonable(jpeg_p),
                    "final_resize": {"mode": final_resize_mode, "target_size": [ori_h // scale_final, ori_w // scale_final]},
                }

        second_rand_first = random.random()
        second_rand_second = random.random()
        if second_rand_first < self.first_order_prob and second_rand_second < self.second_order_prob:
            if return_meta:
                meta["stage"] = "second_order"
            # ----------------------- The first degradation process ----------------------- #
            # blur
            if kernel1.dim() == 2:
                kernel1 = kernel1.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif kernel1.dim() == 3:
                kernel1 = kernel1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif kernel1.dim() == 4 and kernel1.size(0) == 1: # [1, 1, k, k]
                kernel1 = kernel1.repeat(batch_size, 1, 1, 1)

            out = filter2D(out.contiguous(), kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                if return_meta:
                    sigma = float(np.random.uniform(*self.opt['noise_range']))
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=[sigma, sigma], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    noise_meta = {"type": "gaussian", "sigma": sigma, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                if return_meta:
                    scale_noise = float(np.random.uniform(*self.opt['poisson_scale_range']))
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=[scale_noise, scale_noise],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    noise_meta = {"type": "poisson", "scale": scale_noise, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            if return_meta:
                meta["stage1"] = {
                    "blur_applied": True,
                    "kernel": meta["kernel1"],
                    "resize": {"mode": mode, "updown_type": updown_type, "scale": float(scale)},
                    "noise": noise_meta,
                    "jpeg_quality": to_jsonable(jpeg_p),
                }

            # ----------------------- The second degradation process ----------------------- #
            # blur
            second_blur_applied = np.random.uniform() < self.opt['second_blur_prob']
            if second_blur_applied:
                if kernel2.dim() == 2:
                    kernel2 = kernel2.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
                elif kernel2.dim() == 3:
                    kernel2 = kernel2.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                elif kernel2.dim() == 4 and kernel2.size(0) == 1:
                    kernel2 = kernel2.repeat(batch_size, 1, 1, 1)
                out = filter2D(out.contiguous(), kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                if return_meta:
                    sigma = float(np.random.uniform(*self.opt['noise_range2']))
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=[sigma, sigma], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    noise2_meta = {"type": "gaussian", "sigma": sigma, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                if return_meta:
                    scale_noise = float(np.random.uniform(*self.opt['poisson_scale_range2']))
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=[scale_noise, scale_noise],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    noise2_meta = {"type": "poisson", "scale": scale_noise, "gray_prob": gray_noise_prob}
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            if return_meta:
                meta["stage2"] = {
                    "blur_applied": bool(second_blur_applied),
                    "kernel": meta["kernel2"],
                    "resize": {
                        "mode": mode,
                        "updown_type": updown_type,
                        "scale": float(scale),
                        "target_size": [int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)],
                    },
                    "noise": noise2_meta,
                }

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if sinc_kernel.dim() == 2:
                sinc_kernel = sinc_kernel.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif sinc_kernel.dim() == 3:
                sinc_kernel = sinc_kernel.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif sinc_kernel.dim() == 4 and sinc_kernel.size(0) == 1:
                sinc_kernel = sinc_kernel.repeat(batch_size, 1, 1, 1)
            if np.random.uniform() < 0.5:
                final_order = "resize_sinc_then_jpeg"
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
                out = filter2D(out.contiguous(), sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                final_order = "jpeg_then_resize_sinc"
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
                out = filter2D(out.contiguous(), sinc_kernel)
            if return_meta:
                meta["stage2"]["jpeg_quality"] = to_jsonable(jpeg_p)
                meta["stage2"]["final_resize"] = {"mode": mode, "target_size": [ori_h // scale_final, ori_w // scale_final]}
                meta["final_order"] = final_order

        gray_applied = np.random.uniform() < self.opt['gray_prob']
        if gray_applied:
            out = rgb_to_grayscale(out, num_output_channels=3)
        if return_meta:
            meta["gray"] = {"applied": bool(gray_applied), "prob": self.opt['gray_prob']}

        color_jitter_applied = np.random.uniform() < self.opt['color_jitter_prob']
        if color_jitter_applied:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            if return_meta:
                out, color_jitter_meta = self.color_jitter_pt(out, brightness, contrast, saturation, hue, return_meta=True)
                meta["color_jitter"] = color_jitter_meta
            else:
                out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)
        elif return_meta:
            meta["color_jitter"] = {"applied": False, "prob": self.opt['color_jitter_prob']}

        if resize_bak:
            if random.random() > self.second_order_prob and random.random() > self.first_order_prob:
                mode = random.choice(['bicubic'])
            else:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
            if return_meta:
                meta["resize_back"] = {"mode": mode, "target_size": [ori_h, ori_w]}
        # clamp and round
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        if return_meta:
            meta["degradation_vector"] = build_degradation_vector_from_meta(meta)
            return img_gt, img_lq, to_jsonable(meta)
        return img_gt, img_lq
