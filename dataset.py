import os
import random
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