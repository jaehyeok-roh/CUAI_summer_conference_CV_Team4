"""
RD-Curve 생성 스크립트

eval15 전체(15장)에 대해 아래 4개 모델을 동일한 조건(저조도 입력 -> 정상조도 타겟)으로
평가하고 BPP-PSNR / BPP-SSIM 곡선을 그린다.

  - Baseline 1 : 순정 CompressAI (학습 없음), quality 1~8
  - Baseline 2 : 가우시안 노이즈(AWGN)로 학습, quality 2/4/6/8
  - Baseline 3 : LOL 데이터 + 순수 RD Loss, quality 2/4/6/8
  - Ours       : CBAM(decoder) + Edge Loss(20) + TV Loss(40), quality 2/4/6/8

체크포인트 출처:
  - Baseline 2/3 : 로컬에서 학습한 결과를 그대로 사용 (./checkpoints/final_*.pth)
  - Ours         : wandb run 에서 받아 ./ckpt_cache 에 캐싱 (이미 캐싱돼 있으면 재다운로드 안 함)
  - Baseline 1   : 별도 체크포인트 없음, torch hub 에서 compressai 사전학습 가중치만 받음

실행:
    python3 rd_curve.py
"""

import os
import sys
import json

# models.py 는 로컬에서 src/ 아래에 있다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compressai.zoo import bmshj2018_hyperprior
from models import HyperpriorWithCBAM

from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim

CONFIG = {
    "eval_dir": "./LOL_Dataset/lol_dataset/eval15",
    "cache_dir": "./ckpt_cache",
    "local_ckpt_dir": "./checkpoints",  # 로컬에서 직접 학습해서 나온 baseline2/3 체크포인트 위치
    "out_dir": "./results/rd_curve",
    "wandb_entity": "nojh4237-chung-ang-university",
    "wandb_project": "CUAI_summer_Project",
    "baseline1_qualities": [1, 2, 3, 4, 5, 6, 7, 8],
    "qualities": [2, 4, 6, 8],
}

# Ours: edge_weight=20, tv_weight=40 으로 통일된 4개 run
OURS_RUNS = {
    2: "Model_DECODER_Q2_Weight20",
    4: "Model_DECODER_Q4_Weight20",
    6: "Model_DECODER_Q6_Weight20",
    8: "Model_DECODER_Q8_Weight20",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 데이터 로딩
def load_eval_pairs(eval_dir):
    """eval15 의 (low, high) 쌍을 원본 해상도 그대로 텐서로 읽는다."""
    low_dir = os.path.join(eval_dir, "low")
    high_dir = os.path.join(eval_dir, "high")
    names = sorted(os.listdir(low_dir))

    pairs = []
    for name in names:
        low = Image.open(os.path.join(low_dir, name)).convert("RGB")
        high = Image.open(os.path.join(high_dir, name)).convert("RGB")
        pairs.append((TF.to_tensor(low), TF.to_tensor(high), name))
    return pairs


def pad_to_64(x):
    """모델 입력용으로 가로/세로를 64의 배수로 반사 패딩. (dataset.py 평가 모드와 동일)"""
    _, h, w = x.shape
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    if pad_h or pad_w:
        x = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    return x, h, w


# ---------------------------------------------------------------- 평가 루프
@torch.no_grad()
def evaluate(model, pairs):
    """eval15 전체 평균 PSNR / SSIM / BPP 를 계산한다.

    지표는 패딩 영역을 제외한 원본 해상도 기준으로 계산하고,
    BPP 도 원본 픽셀 수로 나눈다.
    """
    model.eval().to(device)
    total_psnr = total_ssim = total_bpp = 0.0

    for low, high, _ in pairs:
        low_p, h, w = pad_to_64(low)
        low_p = low_p.unsqueeze(0).to(device)
        high = high.unsqueeze(0).to(device)

        out = model(low_p)
        # 패딩 영역을 잘라내 원본 해상도로 되돌린 뒤 비교
        x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)

        num_pixels = h * w
        bpp = sum(
            torch.log2(l.clamp(min=1e-9)).sum() / (-num_pixels)
            for l in out["likelihoods"].values()
        ).item()

        total_psnr += compute_psnr(x_hat, high, data_range=1.0).item()
        total_ssim += compute_ssim(x_hat, high, data_range=1.0).item()
        total_bpp += bpp

    n = len(pairs)
    return {"psnr": total_psnr / n, "ssim": total_ssim / n, "bpp": total_bpp / n}


# ---------------------------------------------------------------- 체크포인트
def fetch_from_wandb(run_name, filename="checkpoints/final_decoder_W20.pth"):
    """wandb run 에서 체크포인트를 내려받아 로컬 경로를 돌려준다."""
    import wandb

    cache = os.path.join(CONFIG["cache_dir"], f"{run_name}.pth")
    if os.path.exists(cache):
        return cache

    api = wandb.Api()
    runs = api.runs(f"{CONFIG['wandb_entity']}/{CONFIG['wandb_project']}")
    target = next((r for r in runs if r.name == run_name), None)
    if target is None:
        raise RuntimeError(f"wandb 에서 run 을 찾지 못했습니다: {run_name}")

    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    downloaded = target.file(filename).download(
        root=os.path.join(CONFIG["cache_dir"], run_name), replace=True
    )
    os.replace(downloaded.name, cache)
    return cache




# 엔트로피 코더가 update() 후에만 채우는 버퍼들.
# 실제 비트스트림 인코딩(compress)에만 쓰이고 forward() 기반 BPP 계산에는 불필요하므로,
# 새 모델(빈 버퍼)과 shape 이 달라도 무시하고 넘어간다.
ENTROPY_CODER_BUFFERS = ("_offset", "_quantized_cdf", "_cdf_length", "scale_table")


def load_state_dict_into(model, ckpt_path):
    """체크포인트를 모델에 싣는다. 접두사 차이와 엔트로피 코더 버퍼 불일치를 함께 처리한다."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_state = model.state_dict()

    # 순정 CompressAI 로 저장된 체크포인트라면 base_model. 접두사를 붙여준다.
    if not any(k.startswith("base_model.") for k in state):
        state = {f"base_model.{k}": v for k, v in state.items()}

    filtered, skipped = {}, []
    for k, v in state.items():
        if k not in model_state:
            skipped.append(k)
            continue
        if model_state[k].shape != v.shape:
            if k.endswith(ENTROPY_CODER_BUFFERS):
                skipped.append(k)
                continue
            raise RuntimeError(f"shape 불일치: {k} {tuple(v.shape)} vs {tuple(model_state[k].shape)}")
        filtered[k] = v

    missing, _ = model.load_state_dict(filtered, strict=False)
    # 건너뛴 엔트로피 버퍼 외에 진짜 가중치가 빠졌다면 잘못 로드된 것이다.
    real_missing = [k for k in missing if not k.endswith(ENTROPY_CODER_BUFFERS)]
    if real_missing:
        raise RuntimeError(f"가중치가 누락되었습니다: {real_missing[:5]}")
    return model


# ---------------------------------------------------------------- 모델별 측정
def measure_baseline1(pairs):
    """학습 없는 순정 CompressAI. 저조도 입력을 그대로 압축했을 때의 성능."""
    points = []
    for q in CONFIG["baseline1_qualities"]:
        model = bmshj2018_hyperprior(quality=q, pretrained=True)
        m = evaluate(model, pairs)
        m["quality"] = q
        points.append(m)
        print(f"  Baseline1 q={q}: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")
    return points


def measure_baseline2(pairs):
    """로컬에서 학습한 baseline/baseline2_train.py 결과 (./checkpoints/final_B2_AWGN_Q{q}.pth)."""
    points = []
    for q in CONFIG["qualities"]:
        ckpt = os.path.join(CONFIG["local_ckpt_dir"], f"final_B2_AWGN_Q{q}.pth")
        model = HyperpriorWithCBAM(quality=q, cbam_position="none", pretrained=False)
        load_state_dict_into(model, ckpt)
        m = evaluate(model, pairs)
        m["quality"] = q
        points.append(m)
        print(f"  Baseline2 q={q}: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")
    return points


def measure_baseline3(pairs):
    """로컬에서 학습한 루트 train.py 결과 (edge=0, tv=0, cbam=none) -> ./checkpoints/final_none_Q{q}_W0.pth."""
    points = []
    for q in CONFIG["qualities"]:
        ckpt = os.path.join(CONFIG["local_ckpt_dir"], f"final_none_Q{q}_W0.pth")
        model = HyperpriorWithCBAM(quality=q, cbam_position="none", pretrained=False)
        load_state_dict_into(model, ckpt)
        m = evaluate(model, pairs)
        m["quality"] = q
        points.append(m)
        print(f"  Baseline3 q={q}: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")
    return points


def measure_ours(pairs):
    points = []
    for q, run_name in OURS_RUNS.items():
        ckpt = fetch_from_wandb(run_name)
        model = HyperpriorWithCBAM(quality=q, cbam_position="decoder", pretrained=False)
        load_state_dict_into(model, ckpt)
        m = evaluate(model, pairs)
        m["quality"] = q
        points.append(m)
        print(f"  Ours q={q}: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")
    return points


# ---------------------------------------------------------------- 그래프
# matplotlib 한글 라벨이 깨지므로 영문으로 표기한다.
SERIES_STYLE = {
    "Baseline 1 (No training)": {"color": "#888888", "marker": "o", "ls": "--"},
    "Baseline 2 (AWGN)":        {"color": "#2C7BB6", "marker": "s", "ls": "-"},
    "Baseline 3 (LOL, RD only)": {"color": "#FDAE61", "marker": "^", "ls": "-"},
    "Ours (CBAM+Edge20+TV40)":  {"color": "#D7191C", "marker": "D", "ls": "-"},
}


def plot_curve(results, metric, ylabel, out_path):
    plt.figure(figsize=(7, 5))

    for label, points in results.items():
        pts = sorted(points, key=lambda p: p["bpp"])
        xs = [p["bpp"] for p in pts]
        ys = [p[metric] for p in pts]
        style = SERIES_STYLE[label]

        if len(pts) == 1:
            # 단일 지점(quality 2 에서만 학습된 baseline)은 선 없이 마커로 표시
            plt.plot(xs, ys, marker=style["marker"], color=style["color"],
                     markersize=11, ls="none", label=f"{label} [Q2 only]")
        else:
            plt.plot(xs, ys, marker=style["marker"], color=style["color"],
                     ls=style["ls"], linewidth=1.8, markersize=7, label=label)
            for p in pts:
                plt.annotate(f"Q{p['quality']}", (p["bpp"], p[metric]),
                             textcoords="offset points", xytext=(5, -11), fontsize=8,
                             color=style["color"])

    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel(ylabel)
    plt.title(f"RD Curve on LOL eval15 : BPP vs {ylabel}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"저장됨: {out_path}")


# ---------------------------------------------------------------- 메인
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    pairs = load_eval_pairs(CONFIG["eval_dir"])
    print(f"eval15 이미지 {len(pairs)}장 로드 완료 (device: {device})\n")

    results = {}
    print("[1/4] Baseline 1 (순정 CompressAI, 학습 없음)")
    results["Baseline 1 (No training)"] = measure_baseline1(pairs)

    print("\n[2/4] Baseline 2 (AWGN 학습)")
    results["Baseline 2 (AWGN)"] = measure_baseline2(pairs)

    print("\n[3/4] Baseline 3 (LOL + 순수 RD Loss)")
    results["Baseline 3 (LOL, RD only)"] = measure_baseline3(pairs)

    print("\n[4/4] Ours (CBAM + Edge 20 + TV 40)")
    results["Ours (CBAM+Edge20+TV40)"] = measure_ours(pairs)

    json_path = os.path.join(CONFIG["out_dir"], "rd_points.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n측정값 저장됨: {json_path}")

    plot_curve(results, "psnr", "PSNR (dB)",
               os.path.join(CONFIG["out_dir"], "rd_curve_psnr.png"))
    plot_curve(results, "ssim", "SSIM",
               os.path.join(CONFIG["out_dir"], "rd_curve_ssim.png"))


if __name__ == "__main__":
    main()
