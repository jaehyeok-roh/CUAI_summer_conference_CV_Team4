"""
Baseline 2 (가우시안 JDC) 학습 스크립트

깨끗한 사진(high)에 인공 가우시안 노이즈를 씌워 "노이즈 제거 + 압축"을 함께 학습한다.
저조도(low) 사진은 학습에 쓰지 않으며, 나중에 실제 LOL 저조도 입력을 넣었을 때
밝기 복원에 실패하는 것(Domain Shift)을 보이는 것이 이 baseline 의 목적이다.

Baseline_2.ipynb 의 GaussianNoiseDataset 을 train.py 와 같은 구조의 스크립트로 옮긴 것.

실행:
    python3 baseline/baseline2_train.py
"""

import os
import sys
import random

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import wandb

# models.py 는 레포 루트(또는 로컬의 src/)에 있다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from models import HyperpriorWithCBAM

CONFIG = {
    "train_path": "/content/lol_dataset/lol_dataset/our485",
    "val_path": "/content/lol_dataset/lol_dataset/eval15",
    "save_dir": "./checkpoints",
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 300,
    "warmup_epochs": 5,
    "save_interval": 10,
    "quality": 2,                # <- Q4 / Q6 / Q8 로 바꿔가며 학습
    "noise_sigma": 25 / 255.0,   # 가우시안 노이즈 세기 (표준적인 값)
    "lr": 1e-4,
    "min_lr": 1e-6,
    "aux_lr": 1e-3,
    "lmbda": 0.0035,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim


class GaussianNoiseDataset(Dataset):
    """깨끗한 사진(high)에 가우시안 노이즈를 인공적으로 씌우는 데이터셋.

    Input : 깨끗한 사진 + 가우시안 노이즈
    Target: 깨끗한 사진
    """

    def __init__(self, root_dir, crop_size=256, noise_sigma=25 / 255.0, is_train=True):
        self.high_dir = os.path.join(root_dir, "high")
        self.image_names = sorted(os.listdir(self.high_dir))
        self.crop_size = crop_size
        self.noise_sigma = noise_sigma
        self.is_train = is_train

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        clean = Image.open(os.path.join(self.high_dir, self.image_names[idx])).convert("RGB")
        w, h = TF.get_image_size(clean)

        if self.is_train:
            crop_i = random.randint(0, h - self.crop_size)
            crop_j = random.randint(0, w - self.crop_size)
            clean = TF.crop(clean, crop_i, crop_j, self.crop_size, self.crop_size)
            if random.random() > 0.5:
                clean = TF.hflip(clean)
        else:
            # 평가 시에는 크롭 없이 64의 배수로 반사 패딩 (dataset.py 평가 모드와 동일)
            pad_w = (64 - (w % 64)) % 64
            pad_h = (64 - (h % 64)) % 64
            if pad_w or pad_h:
                clean = TF.pad(clean, (0, 0, pad_w, pad_h), padding_mode="reflect")

        clean = TF.to_tensor(clean)
        # 노이즈는 매번 새로 생성하되, 0~1 범위를 벗어나지 않게 자른다.
        noisy = (clean + torch.randn_like(clean) * self.noise_sigma).clamp(0, 1)
        return noisy, clean


def freeze_base_model(model, freeze=True):
    for name, param in model.named_parameters():
        if "cbam" not in name:
            param.requires_grad = not freeze


def rate_distortion_loss(out_net, target, lmbda):
    """Baseline 2 는 Edge/TV 없이 순수 RD Loss 만 사용한다."""
    N, _, H, W = target.size()
    num_pixels = N * H * W

    bpp = sum(
        (torch.log2(l.clamp(min=1e-9)).sum() / (-num_pixels))
        for l in out_net["likelihoods"].values()
    )
    mse = torch.nn.functional.mse_loss(out_net["x_hat"], target)
    distortion = lmbda * (255 ** 2) * mse
    return bpp + distortion, {"bpp": bpp.item(), "distortion_term": distortion.item()}


def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    wandb.init(
        entity="nojh4237-chung-ang-university",
        project="CUAI_summer_Project",
        name=f"BASELINE2_AWGN_Q{CONFIG['quality']}",
        config=CONFIG,
    )

    train_dataset = GaussianNoiseDataset(
        CONFIG["train_path"], crop_size=256,
        noise_sigma=CONFIG["noise_sigma"], is_train=True,
    )
    val_dataset = GaussianNoiseDataset(
        CONFIG["val_path"], crop_size=256,
        noise_sigma=CONFIG["noise_sigma"], is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    # CBAM 없는 순정 구조. rd_curve.py 가 같은 클래스로 체크포인트를 읽을 수 있게 맞춘다.
    model = HyperpriorWithCBAM(
        quality=CONFIG["quality"], cbam_position="none", pretrained=True
    ).to(device)

    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]

    optimizer = optim.Adam(params, lr=CONFIG["lr"])
    aux_optimizer = optim.Adam(aux_params, lr=CONFIG["aux_lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])

    print(f"Setup Completed - Baseline 2 (AWGN) Q{CONFIG['quality']} on {device}")

    for epoch in range(CONFIG["epochs"]):
        is_warmup = epoch < CONFIG["warmup_epochs"]
        freeze_base_model(model, freeze=is_warmup)

        # --- TRAIN ---
        model.train()
        train_loss, train_bpp, train_mse = 0.0, 0.0, 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            out_net = model(noisy)
            loss, logs = rate_distortion_loss(out_net, clean, CONFIG["lmbda"])
            loss.backward()
            optimizer.step()

            if not is_warmup:
                model.aux_loss().backward()
                aux_optimizer.step()

            train_loss += loss.item()
            train_bpp += logs["bpp"]
            train_mse += logs["distortion_term"]

        scheduler.step()
        steps = len(train_loader)

        # --- VALIDATION (노이즈 제거 성능) ---
        model.eval()
        val_loss, val_bpp, val_psnr, val_ssim = 0.0, 0.0, 0.0, 0.0
        logged_images = False

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                out_net = model(noisy)
                loss, logs = rate_distortion_loss(out_net, clean, CONFIG["lmbda"])
                x_hat = out_net["x_hat"].clamp(0, 1)

                val_loss += loss.item()
                val_bpp += logs["bpp"]
                val_psnr += compute_psnr(x_hat, clean, data_range=1.0).item()
                val_ssim += compute_ssim(x_hat, clean, data_range=1.0).item()

                if not logged_images:
                    wandb.log({
                        "Visuals/1_Noisy_Input": wandb.Image(noisy[0].cpu(), caption="Noisy (Input)"),
                        "Visuals/2_B2_Recon": wandb.Image(x_hat[0].cpu(), caption="Reconstructed (Baseline 2)"),
                        "Visuals/3_Clean_Target": wandb.Image(clean[0].cpu(), caption="Clean (Target)"),
                    }, commit=False)
                    logged_images = True

        v_steps = len(val_loader)
        avg_psnr = val_psnr / v_steps

        mode_str = "[WARMUP]" if is_warmup else "[TRAIN]"
        print(f"Epoch {mode_str} [{epoch+1:03d}/{CONFIG['epochs']}] "
              f"Loss [T/V]: {train_loss/steps:.4f}/{val_loss/v_steps:.4f} | PSNR: {avg_psnr:.2f}")

        wandb.log({
            "Train/Loss": train_loss / steps,
            "Train/BPP": train_bpp / steps,
            "Train/MSE_Term": train_mse / steps,
            "Train/LR": optimizer.param_groups[0]["lr"],
            "Val/Loss": val_loss / v_steps,
            "Val/BPP": val_bpp / v_steps,
            "Val/PSNR": avg_psnr,
            "Val/SSIM": val_ssim / v_steps,
            "Epoch": epoch + 1,
        })

        if (epoch + 1) % CONFIG["save_interval"] == 0:
            ckpt_path = os.path.join(
                CONFIG["save_dir"], f"ckpt_B2_AWGN_Q{CONFIG['quality']}_ep{epoch+1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            print(f"Checkpoint saved at Epoch {epoch+1}")

    final_path = os.path.join(CONFIG["save_dir"], f"final_B2_AWGN_Q{CONFIG['quality']}.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    print("Training Completed and Final Model Saved.")

    wandb.finish()


if __name__ == "__main__":
    main()
