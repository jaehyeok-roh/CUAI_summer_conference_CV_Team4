import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from compressai.zoo import bmshj2018_hyperprior
from models import CBAM
from loss import RateDistortionEdgeLoss, calibrate_edge_weight
from dataset import LOLDataset

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "dataset_path": "/workspace/data/lol_dataset/our485",  # 서버의 데이터셋 경로
    "save_dir": "./checkpoints",
    "batch_size": 128,  # find_max_batch.py 테스트 결과(최대 204) 기준 안전 마진 적용
    "num_workers": 4,
    "epochs": 100,
    "quality": 2,          # CompressAI 타겟 퀄리티
    "target_ratio": 0.0,  # Edge 가중치 타겟 비중 (0%)
    "cbam_position": "none",  # Baseline 3: CBAM 없음 ("encoder", "decoder", "none" 중 하나)
    "lr": 1e-4,
    "aux_lr": 1e-3,
    "lmbda": 0.0035,
    "eval_interval": 10    # 자동 저장(체크포인트) 간격
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # 1. WandB 초기화 및 연동 (프로젝트명과 실험 이름)
    # cbam_position=none, target_ratio=0.0 -> Baseline 3 (순정 CompressAI + LOL, CBAM/Edge Loss 없음)
    variant = "BASELINE3" if CONFIG["cbam_position"] == "none" else "OURS"
    wandb.init(
        project="CUAI_summer_Project",
        name=f"{variant}_{CONFIG['cbam_position'].upper()}_Q{CONFIG['quality']}_Ratio{int(CONFIG['target_ratio']*100)}",
        config=CONFIG
    )

    # 2. 데이터셋 로드
    train_dataset = LOLDataset(root_dir=CONFIG["dataset_path"], crop_size=256)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )

    # 3. 모델 세팅 & CBAM 이식
    model = bmshj2018_hyperprior(quality=CONFIG["quality"], pretrained=True).to(device)

    if CONFIG["cbam_position"] == "encoder":
        cbam_module = CBAM(in_channels=model.M, reduction=16).to(device)
        encoder_layers = list(model.g_a.children())
        encoder_layers.append(cbam_module)
        model.g_a = nn.Sequential(*encoder_layers)
    elif CONFIG["cbam_position"] == "decoder":
        cbam_module = CBAM(in_channels=model.M, reduction=16).to(device)
        decoder_layers = list(model.g_s.children())
        decoder_layers.insert(0, cbam_module)
        model.g_s = nn.Sequential(*decoder_layers)
    elif CONFIG["cbam_position"] == "none":
        pass  # Baseline 3: CBAM 없이 순정 CompressAI 구조 그대로 사용
    else:
        raise ValueError("cbam_position config must be 'encoder', 'decoder', or 'none'")

    model.train()

    # 4. 손실 함수 (Edge Loss 비율 자동 계산)
    if CONFIG["target_ratio"] == 0.0:
        # Baseline 3: Edge Loss 비중 0 -> 캘리브레이션 없이 바로 0으로 고정 (순수 BPP+MSE)
        optimal_gamma = 0.0
    else:
        dummy_criterion = RateDistortionEdgeLoss(lmbda=CONFIG["lmbda"], edge_weight=0.1, mse_blur_sigma=0.0).to(device)
        optimal_gamma = calibrate_edge_weight(
            model=model, loader=train_loader, criterion=dummy_criterion,
            target_ratio=CONFIG["target_ratio"], n_batches=5, device=device, verbose=False
        )

    wandb.config.update({"optimal_gamma": optimal_gamma}) # 찾아낸 가중치값도 WandB 기록
    criterion = RateDistortionEdgeLoss(lmbda=CONFIG["lmbda"], edge_weight=optimal_gamma, mse_blur_sigma=0.0).to(device)

    # 5. 최적화 도구 세팅
    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = optim.Adam(params, lr=CONFIG["lr"])
    aux_optimizer = optim.Adam(aux_params, lr=CONFIG["aux_lr"])

    print(f"Training started on {device} (Epochs: {CONFIG['epochs']})")

    # 6. 메인 학습 루프
    for epoch in range(CONFIG["epochs"]):
        total_loss, total_bpp, total_mse, total_edge = 0.0, 0.0, 0.0, 0.0
        
        for low_img, high_img in train_loader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            out_net = model(low_img)
            loss, logs = criterion(out_net, high_img)
            
            loss.backward()
            optimizer.step()
            
            model.aux_loss().backward()
            aux_optimizer.step()
            
            # 메트릭 합산
            total_loss += loss.item()
            total_bpp += logs['bpp']
            total_mse += logs['distortion_term']
            total_edge += logs['edge_term']
            
        # 에포크 당 평균값 계산
        steps = len(train_loader)
        avg_loss = total_loss / steps
        avg_bpp = total_bpp / steps
        avg_mse = total_mse / steps
        avg_edge = total_edge / steps

        # 터미널용 1줄 로그
        print(f"Epoch [{epoch+1:03d}/{CONFIG['epochs']}] "
              f"Loss: {avg_loss:.4f} | BPP: {avg_bpp:.4f} | MSE_term: {avg_mse:.4f} | Edge_term: {avg_edge:.4f}")

        # WandB 로 실시간 전송
        wandb.log({
            "Train/Loss_Total": avg_loss,
            "Train/BPP": avg_bpp,
            "Train/MSE_Term": avg_mse,
            "Train/Edge_Term": avg_edge,
            "Epoch": epoch + 1
        })

        # 가중치 자동 저장(CheckPoint)
        if (epoch + 1) % CONFIG["eval_interval"] == 0:
            ckpt_name = f"{variant.lower()}_{CONFIG['cbam_position']}_Q{CONFIG['quality']}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], ckpt_name))

    # 최종 가중치 저장 및 종료
    final_path = os.path.join(CONFIG["save_dir"], f"{variant.lower()}_FINAL_{CONFIG['cbam_position']}_Q{CONFIG['quality']}_Ratio{int(CONFIG['target_ratio']*100)}.pth")
    torch.save(model.state_dict(), final_path)
    wandb.finish()

if __name__ == "__main__":
    main()