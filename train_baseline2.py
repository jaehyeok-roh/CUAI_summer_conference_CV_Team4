import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import wandb

from models import HyperpriorWithCBAM
from loss import RateDistortionEdgeLoss

class GaussianNoiseDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, noise_sigma=25/255.0, is_train=True):
        self.high_dir = os.path.join(root_dir, 'high')  # ★ 오직 밝고 깨끗한 사진(high)만 봅니다!
        self.image_names = sorted(os.listdir(self.high_dir))
        self.crop_size = crop_size
        self.noise_sigma = noise_sigma
        self.is_train = is_train

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        clean_img = Image.open(os.path.join(self.high_dir, img_name)).convert('RGB')
        w, h = TF.get_image_size(clean_img)
        
        # [패딩 방어선] 
        if w < self.crop_size or h < self.crop_size:
            pad_w = max(0, self.crop_size - w)
            pad_h = max(0, self.crop_size - h)
            clean_img = TF.pad(clean_img, (0, 0, pad_w, pad_h), padding_mode='reflect')
            w, h = TF.get_image_size(clean_img)

        # 자르기 및 증강
        if self.is_train:
            crop_i = random.randint(0, h - self.crop_size)
            crop_j = random.randint(0, w - self.crop_size)
            clean_img = TF.crop(clean_img, crop_i, crop_j, self.crop_size, self.crop_size)
            if random.random() > 0.5:
                clean_img = TF.hflip(clean_img)
        else:
            crop_i = (h - self.crop_size) // 2
            crop_j = (w - self.crop_size) // 2
            clean_img = TF.crop(clean_img, crop_i, crop_j, self.crop_size, self.crop_size)

        clean_tensor = TF.to_tensor(clean_img)

        noise = torch.randn_like(clean_tensor) * self.noise_sigma
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)

        return noisy_tensor, clean_tensor

# ==========================================
# 1. B2 Configuration 
# ==========================================
CONFIG = {
    "train_path": "/workspace/data/lol_dataset/our485", 
    "val_path": "/workspace/data/lol_dataset/eval15",
    "save_dir": "./checkpoints",
    "batch_size": 32,             
    "num_workers": 4,
    "epochs": 100, 
    "warmup_epochs": 0,          # (순정모델이므로 Warm-up 생략)
    "save_interval": 10,         
    "quality": 2,          
    "target_ratio": 0.00,        # ★ B2는 Edge Loss 없이 MSE로만 (기존 논문 구현)
    "cbam_position": "none",     # ★ B2는 CBAM 안 붙입니다
    "lr": 1e-4,                   
    "min_lr": 1e-6,               
    "aux_lr": 1e-3,
    "lmbda": 0.0035,
    "noise_sigma": 25/255.0
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 지표 검증 모듈
try:
    from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim
except ImportError:
    pass

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    wandb.init(
        entity="nojh4237-chung-ang-university",
        project="CUAI_summer_Project",
        name=f"BASELINE_2_AWGN_Q{CONFIG['quality']}",
        config=CONFIG
    )

    train_dataset = GaussianNoiseDataset(root_dir=CONFIG["train_path"], crop_size=256, noise_sigma=CONFIG["noise_sigma"], is_train=True)
    val_dataset = GaussianNoiseDataset(root_dir=CONFIG["val_path"], crop_size=256, noise_sigma=CONFIG["noise_sigma"], is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = HyperpriorWithCBAM(quality=CONFIG["quality"], cbam_position=CONFIG["cbam_position"], pretrained=True).to(device)

    criterion = RateDistortionEdgeLoss(lmbda=CONFIG["lmbda"], edge_weight=0.0, mse_blur_sigma=0.0).to(device)

    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = optim.Adam(params, lr=CONFIG["lr"])
    aux_optimizer = optim.Adam(aux_params, lr=CONFIG["aux_lr"])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])
    aux_scheduler = CosineAnnealingLR(aux_optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])

    print(f"Baseline 2 Setup Completed - Starting Run on {device} (Epochs: {CONFIG['epochs']})")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss, train_bpp, train_mse = 0.0, 0.0, 0.0
        
        for low_img, high_img in train_loader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            optimizer.zero_grad(); aux_optimizer.zero_grad()
            
            out_net = model(low_img)
            loss, logs = criterion(out_net, high_img)
            loss.backward()
            optimizer.step()
            model.aux_loss().backward()
            aux_optimizer.step()
            
            train_loss += loss.item()
            train_bpp += logs['bpp']
            train_mse += logs['distortion_term']
            
        scheduler.step()
        aux_scheduler.step()
        steps = len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss, val_bpp, val_psnr, val_ssim = 0.0, 0.0, 0.0, 0.0
        logged_images = False

        with torch.no_grad():
            for low_img, high_img in val_loader:
                low_img, high_img = low_img.to(device), high_img.to(device)
                
                out_net = model(low_img)
                loss, logs = criterion(out_net, high_img)
                x_hat = out_net['x_hat'].clamp(0, 1)

                val_loss += loss.item()
                val_bpp += logs['bpp']
                val_psnr += compute_psnr(x_hat, high_img, data_range=1.0).item()
                val_ssim += compute_ssim(x_hat, high_img, data_range=1.0).item()

                if not logged_images:
                    wandb.log({
                        "Visuals/1_Gaussian_Input": wandb.Image(low_img[0].cpu(), caption="Synthetic Noise (AWGN)"),
                        "Visuals/2_B2_Recon": wandb.Image(x_hat[0].cpu(), caption="Reconstructed (Baseline 2)"),
                        "Visuals/3_High_Target": wandb.Image(high_img[0].cpu(), caption="High Light (Clean Target)")
                    }, commit=False)  
                    logged_images = True

        v_steps = len(val_loader)
        
        print(f"Epoch [{epoch+1:03d}/{CONFIG['epochs']}] Train/Val Loss: {train_loss/steps:.4f}/{val_loss/v_steps:.4f} | Val PSNR: {val_psnr/v_steps:.2f}")
        
        wandb.log({
            "Train/Loss": train_loss / steps, "Train/BPP": train_bpp / steps, "Train/LR": optimizer.param_groups[0]['lr'],
            "Val/Loss": val_loss / v_steps, "Val/BPP": val_bpp / v_steps, "Val/PSNR": val_psnr / v_steps, "Val/SSIM": val_ssim / v_steps, "Epoch": epoch + 1
        })

        if (epoch + 1) % CONFIG["save_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"ckpt_B2_AWGN_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            
    final_path = os.path.join(CONFIG["save_dir"], f"final_B2_AWGN.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    print("Baseline 2 Training Completed and Final Model Saved.")
    wandb.finish()

if __name__ == "__main__":
    main()