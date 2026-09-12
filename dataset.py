import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class LOLDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, is_train=True):
        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')
        self.image_names = sorted(os.listdir(self.low_dir))
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        low_img = Image.open(os.path.join(self.low_dir, img_name)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, img_name)).convert('RGB')

        w, h = TF.get_image_size(low_img)

        if self.is_train:
            # [방어적 패딩] 이미지 크기가 crop_size 보다 작을 경우 반사(reflect) 패딩 처리
            if w < self.crop_size or h < self.crop_size:
                pad_w = max(0, self.crop_size - w)
                pad_h = max(0, self.crop_size - h)
                low_img = TF.pad(low_img, (0, 0, pad_w, pad_h), padding_mode='reflect')
                high_img = TF.pad(high_img, (0, 0, pad_w, pad_h), padding_mode='reflect')
                w, h = TF.get_image_size(low_img)

            crop_i = random.randint(0, h - self.crop_size)
            crop_j = random.randint(0, w - self.crop_size)
            low_img = TF.crop(low_img, crop_i, crop_j, self.crop_size, self.crop_size)
            high_img = TF.crop(high_img, crop_i, crop_j, self.crop_size, self.crop_size)
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)
        else:
            # [평가 모드 패딩] BPP 및 PSNR 정밀 측정을 위해 크롭을 배제하고 64의 배수로 반사 패딩
            pad_w = (64 - (w % 64)) % 64
            pad_h = (64 - (h % 64)) % 64
            
            if pad_w > 0 or pad_h > 0:
                low_img = TF.pad(low_img, (0, 0, pad_w, pad_h), padding_mode='reflect')
                high_img = TF.pad(high_img, (0, 0, pad_w, pad_h), padding_mode='reflect')

        return TF.to_tensor(low_img), TF.to_tensor(high_img)


class SyntheticLowLight(LOLDataset):
    """정상조도(high) 이미지로 저조도 입력을 합성한다. low 이미지는 어둡게 할 정도를 정하는 데만 쓴다.

    mode:
      dark      : 노출만 낮춘다 (노이즈 없음)
      dark_pg   : 노출을 낮추고 신호에 비례하는 Poisson-Gaussian 노이즈를 더한다
                  (노이즈 레벨 샘플링은 Brooks et al., CVPR 2019 "Unprocessing Images for Learned Raw Denoising")
      dark_awgn : 노출을 낮추고 dark_pg 와 이미지 평균 분산이 같은 AWGN 을 더한다 (노이즈 '형태'만 다름)
    """
    MODES = ("dark", "dark_awgn", "dark_pg")

    def __init__(self, root_dir, mode, crop_size=256):
        super().__init__(root_dir, crop_size, is_train=True)
        if mode not in self.MODES:
            raise ValueError(f"mode 는 {self.MODES} 중 하나여야 합니다: {mode}")
        self.mode = mode
        # 실제 LOL 쌍의 밝기 비율(low 평균 / high 평균)을 이미지별로 그대로 쓴다
        self.ratios = [
            np.asarray(Image.open(os.path.join(self.low_dir, n)).convert('RGB'), dtype=np.float64).mean()
            / np.asarray(Image.open(os.path.join(self.high_dir, n)).convert('RGB'), dtype=np.float64).mean()
            for n in self.image_names
        ]

    def __getitem__(self, idx):
        _, high = super().__getitem__(idx)  # LOLDataset 과 같은 크롭/반전이 적용된 high
        # ponytail: sRGB 감마를 2.2 거듭제곱으로 근사, 정확한 sRGB 곡선이 필요하면 교체
        lin = (high * self.ratios[idx]) ** 2.2
        if self.mode != "dark":
            log_shot = random.uniform(math.log(1e-4), math.log(0.012))
            shot = math.exp(log_shot)
            read = math.exp(2.18 * log_shot + 1.20 + random.gauss(0, 0.26))
            var = lin * shot + read if self.mode == "dark_pg" else lin.mean() * shot + read
            lin = lin + torch.randn_like(lin) * var.sqrt()
        return lin.clamp(0, 1) ** (1 / 2.2), high