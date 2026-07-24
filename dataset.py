import sys
import os
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from compressai.zoo import bmshj2018_hyperprior

try:
    from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
except ImportError:
    print("torchmetrics가 없습니다. '!pip install torchmetrics'를 먼저 실행해주세요.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 중인 연산 장치: {device}")


class LOLDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')

        self.image_names = sorted(os.listdir(self.low_dir))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. 이미지 불러오기
        img_name = self.image_names[idx]
        low_img = Image.open(os.path.join(self.low_dir, img_name)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, img_name)).convert('RGB')

        # 2. 256x256 크기로 무작위로 자를 위치(좌표) 계산
        w, h = TF.get_image_size(low_img)
        crop_i = random.randint(0, h - self.crop_size) # 세로 자를 위치
        crop_j = random.randint(0, w - self.crop_size) # 가로 자를 위치

        # 3. 계산된 똑같은 위치로 low와 high 이미지를 동시에 자르기
        low_img = TF.crop(low_img, crop_i, crop_j, self.crop_size, self.crop_size)
        high_img = TF.crop(high_img, crop_i, crop_j, self.crop_size, self.crop_size)

        # 4. 50% 확률로 좌우 반전(Data Augmentation) 똑같이 적용
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        # 5. 텐서(0~1 사이 숫자)로 변환
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        return low_tensor, high_tensor