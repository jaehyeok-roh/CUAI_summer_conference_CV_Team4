import os
import random
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb

from dataset import LOLDataset, SyntheticLowLight
from models import HyperpriorWithCBAM
from loss import RateDistortionEdgeLoss

# 환경 구성 설정
CONFIG = {
    "train_path": "/workspace/data/lol_dataset/our485", 
    "val_path": "/workspace/data/lol_dataset/eval15",
    "save_dir": "./checkpoints",
    "batch_size": 32,             
    "num_workers": 4,
    "epochs": 300,                
    "warmup_epochs": 5,          # 초반 압축기 보호용 가중치 동결 에포크
    "save_interval": 10,         # 정기 체크포인트 저장 주기
    "quality": 2,          
    "edge_weight": 10.0,          # Baseline 3 훈련시 0.0 으로 세팅      
    "tv_weight": 40.0,
    "mse_blur_sigma": 0.0,       # 어긋남 방지를 위한 MSE 블러 적용 (비활성화 시 0.0)
    "cbam_position": "decoder",  # Baseline 3 훈련시 "none" 으로 세팅
    "lr": 1e-4,                   
    "min_lr": 1e-6,               
    "aux_lr": 1e-3,              # aux_lr 은 스케줄러 없이 1e-3 으로 고정
    "lmbda": 0.0035,
    "seed": 0,
    "tag": "",                   # 실행 이름. 지정하면 wandb 이름과 체크포인트 파일명에 사용
    "synthetic": "",             # "" = 실제 LOL 쌍, dark | dark_awgn | dark_pg = high 이미지로 저조도 입력 합성
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 지표 검증 모듈
try:
    from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim
except ImportError:
    raise ImportError("torchmetrics 설치가 필요합니다.")

def freeze_base_model(model, freeze=True):
    """ CompressAI 베이스라인 파라미터의 requires_grad 상태를 제어 """
    for name, param in model.named_parameters():
        if "cbam" not in name:
            param.requires_grad = not freeze

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])  # DataLoader 워커의 random/torch 시드도 여기서 파생된다
    run_name = CONFIG["tag"] or f"Model_{CONFIG['cbam_position'].upper()}_Q{CONFIG['quality']}_Weight{CONFIG['edge_weight']:g}"
    
    wandb.init(
        entity="nojh4237-chung-ang-university",
        project="CUAI_summer_Project",
        name=run_name,
        config=CONFIG
    )

    if CONFIG["synthetic"]:
        train_dataset = SyntheticLowLight(root_dir=CONFIG["train_path"], mode=CONFIG["synthetic"], crop_size=256)
    else:
        train_dataset = LOLDataset(root_dir=CONFIG["train_path"], crop_size=256, is_train=True)
    val_dataset = LOLDataset(root_dir=CONFIG["val_path"], crop_size=256, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = HyperpriorWithCBAM(
        quality=CONFIG["quality"], 
        cbam_position=CONFIG["cbam_position"], 
        pretrained=True
    ).to(device)

    # 정적 가중치와 시그마 값이 적용된 Loss 모듈 초기화
    criterion = RateDistortionEdgeLoss(
        lmbda=CONFIG["lmbda"], 
        edge_weight=CONFIG["edge_weight"], 
        mse_blur_sigma=CONFIG["mse_blur_sigma"],
        tv_weight=CONFIG["tv_weight"],       # [신규 추가]
    ).to(device)

    # 스케줄러 & 옵티마이저 정의
    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    
    optimizer = optim.Adam(params, lr=CONFIG["lr"])
    aux_optimizer = optim.Adam(aux_params, lr=CONFIG["aux_lr"])
    
    # 메인 파라미터에만 스케줄러 적용
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["min_lr"])

    print(f"Setup Completed - Starting Run on {device} (Limit Epochs: {CONFIG['epochs']})")

    for epoch in range(CONFIG["epochs"]):
        # CBAM 이 없으면 동결 시 학습할 파라미터가 없어 backward 에서 에러가 나므로 warmup 을 건너뛴다
        is_warmup = epoch < CONFIG["warmup_epochs"] and model.cbam is not None
        freeze_base_model(model, freeze=is_warmup)
        
        # --- TRAIN ---
        model.train()
        train_loss, train_bpp, train_mse, train_edge, train_tv = 0.0, 0.0, 0.0, 0.0, 0.0  # train_tv 추가
        
        for low_img, high_img in train_loader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            optimizer.zero_grad(); aux_optimizer.zero_grad()
            
            out_net = model(low_img)
            loss, logs = criterion(out_net, high_img)
            loss.backward()
            optimizer.step()
            
            if not is_warmup:
                model.aux_loss().backward()
                aux_optimizer.step()
            
            train_loss += loss.item()
            train_bpp += logs['bpp']
            train_mse += logs['distortion_term']
            train_edge += logs['edge_term']
            train_tv += logs['tv_term']  # TV term 누적
            
        scheduler.step()
        
        steps = len(train_loader)
        avg_train_loss = train_loss / steps

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
                    idx = 0  
                    wandb.log({
                        "Visuals/1_Low_Input": wandb.Image(low_img[idx].cpu(), caption="Low Light (Input)"),
                        "Visuals/2_Ours_Recon": wandb.Image(x_hat[idx].cpu(), caption="Reconstructed (Ours)"),
                        "Visuals/3_High_Target": wandb.Image(high_img[idx].cpu(), caption="High Light (Target)")
                    }, commit=False)  
                    logged_images = True

        v_steps = len(val_loader)
        avg_val_loss = val_loss / v_steps
        avg_psnr = val_psnr / v_steps
        
        mode_str = "[WARMUP]" if is_warmup else "[TRAIN]"
        print(f"Epoch {mode_str} [{epoch+1:03d}/{CONFIG['epochs']}] Loss [T/V]: {avg_train_loss:.4f}/{avg_val_loss:.4f} | PSNR: {avg_psnr:.2f}")
        
        wandb.log({
            "Train/Loss": avg_train_loss, 
            "Train/BPP": train_bpp / steps, 
            "Train/Edge_Term": train_edge / steps,  # Edge Loss 기록 추가
            "Train/TV_Term": train_tv / steps,      # TV Loss 기록 추가
            "Train/MSE_Term": train_mse / steps,    # MSE 항 기록 추가
            "Train/LR": optimizer.param_groups[0]['lr'],
            "Val/Loss": avg_val_loss, 
            "Val/BPP": val_bpp / v_steps, 
            "Val/PSNR": avg_psnr, 
            "Val/SSIM": val_ssim / v_steps, 
            "Epoch": epoch + 1
        })

        # --- REGULAR CHECKPOINT SAVING ---
        if (epoch + 1) % CONFIG["save_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"ckpt_{run_name}_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            print(f"Checkpoint saved at Epoch {epoch+1}")
            
    final_path = os.path.join(CONFIG["save_dir"], f"final_{run_name}.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    print("Training Completed and Final Model Saved.")
                
    wandb.finish()

if __name__ == "__main__":
    # 모든 CONFIG 키를 --키 값 으로 덮어쓸 수 있다 (예: --quality 4 --lmbda 0.013 --tag OURS_Q4)
    parser = argparse.ArgumentParser()
    for key, value in CONFIG.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    CONFIG.update(vars(parser.parse_args()))
    main()
