"""
RD-Curve 생성 스크립트

eval15 전체(15장)에 대해 저조도 입력 -> 정상조도 타겟 조건으로 모델을 평가한다.

RD 곡선(BPP-PSNR / BPP-SSIM)과 Ours vs Baseline 3 BD-rate:
  - Baseline 1 : 순정 CompressAI (학습 없음), quality 1~8
  - Baseline 2 : 가우시안 노이즈(AWGN)로 학습, quality 2
  - Baseline 3 : LOL 데이터 + 순수 RD Loss, quality 2/4/6/8
  - Ours       : CBAM(decoder) + Edge Loss + TV Loss, quality 2/4/6/8
                 (Q4/6/8 은 quality 별 lmbda 로 재학습, edge/tv 가중치는 Q2 의 20/40 에 lmbda 비율을 곱함)
Q2 비교 표(tables_q2.json): ablation, 합성 저조도 학습 데이터

체크포인트 출처:
  - Baseline 1 : 별도 체크포인트 없음, torch hub 에서 compressai 사전학습 가중치만 받음
  - 나머지 전부 : wandb run 에서 받아 ./ckpt_cache 에 캐싱 (아직 끝나지 않은 run 은 건너뜀)

실행:
    python3 rd_curve.py --eval_dir <LOL 경로>/eval15
"""

import os
import sys
import json
import argparse
import functools

# models.py 는 로컬에서 src/ 아래에 있다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
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
    "out_dir": "./results/rd_curve",
    "wandb_entity": "nojh4237-chung-ang-university",
    "wandb_project": "CUAI_summer_Project",
    "baseline1_qualities": [1, 2, 3, 4, 5, 6, 7, 8],
    "qualities": [2, 4, 6, 8],
}

# 평가할 run: (quality, wandb run 이름, wandb 체크포인트 파일, cbam 위치, 같은 이름 run 이 여럿일 때 구분할 config)
def tagged(quality, tag, cbam):
    """train.py --tag 로 학습한 run"""
    return (quality, tag, f"checkpoints/final_{tag}.pth", cbam, {})


B2_RUNS = {2: (2, "BASELINE2_AWGN_Q2", "checkpoints/final_B2_AWGN_Q2.pth", "none", {})}
B3_RUNS = {
    2: (2, "Model_NONE_Q2_Weight0", "checkpoints/final_none_Q2_W0.pth", "none", {"edge_weight": 0, "tv_weight": 0}),
    4: tagged(4, "B3_Q4", "none"),
    6: tagged(6, "B3_Q6", "none"),
    8: tagged(8, "B3_Q8", "none"),
}
OURS_RUNS = {
    2: (2, "Model_DECODER_Q2_Weight20", "checkpoints/final_decoder_W20.pth", "decoder", {"edge_weight": 20, "tv_weight": 40}),
    4: tagged(4, "OURS_Q4", "decoder"),
    6: tagged(6, "OURS_Q6", "decoder"),
    8: tagged(8, "OURS_Q8", "decoder"),
}
ABLATION_RUNS = {
    "Ours": OURS_RUNS[2],
    "w/o CBAM": tagged(2, "ABL_noCBAM", "none"),
    "w/o Edge": tagged(2, "ABL_noEdge", "decoder"),
    "w/o TV": tagged(2, "ABL_noTV", "decoder"),
}
SYNTHETIC_RUNS = {
    "Normal + AWGN (Baseline 2)": B2_RUNS[2],
    "Dark": tagged(2, "SYN_dark", "none"),
    "Dark + AWGN": tagged(2, "SYN_awgn", "none"),
    "Dark + Poisson-Gaussian": tagged(2, "SYN_pg", "none"),
    "Real LOL (Baseline 3)": B3_RUNS[2],
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
@functools.lru_cache(maxsize=None)
def project_runs():
    import wandb
    return list(wandb.Api().runs(f"{CONFIG['wandb_entity']}/{CONFIG['wandb_project']}"))


def fetch_from_wandb(run_name, filename, match):
    """wandb run 에서 체크포인트를 내려받아 로컬 경로를 돌려준다. 조건에 맞는 끝난 run 이 없으면 None."""

    cache = os.path.join(CONFIG["cache_dir"], f"{run_name}.pth")
    if os.path.exists(cache):
        return cache

    target = next((
        r for r in project_runs()
        if r.name == run_name and r.state == "finished"
        and all(r.config.get(k) == v for k, v in match.items())
        and filename in {f.name for f in r.files()}
    ), None)
    if target is None:
        return None

    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    downloaded = target.file(filename).download(
        root=os.path.join(CONFIG["cache_dir"], run_name), replace=True
    )
    downloaded.close()  # Windows 에서는 열린 파일을 옮길 수 없다
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


def measure(label, runs, pairs):
    """runs 의 체크포인트를 차례로 평가한다. 아직 끝나지 않은 run 은 건너뛴다."""
    points = []
    for key, (q, run_name, filename, cbam, match) in runs.items():
        ckpt = fetch_from_wandb(run_name, filename, match)
        if ckpt is None:
            print(f"  {label} [{key}] {run_name}: 끝난 run 이 없어 건너뜀")
            continue
        model = HyperpriorWithCBAM(quality=q, cbam_position=cbam, pretrained=False)
        load_state_dict_into(model, ckpt)
        m = evaluate(model, pairs)
        m.update(quality=q, row=str(key))
        points.append(m)
        print(f"  {label} [{key}]: PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.4f} / BPP {m['bpp']:.4f}")
    return points


# ---------------------------------------------------------------- 그래프
# matplotlib 한글 라벨이 깨지므로 영문으로 표기한다.
SERIES_STYLE = {
    "Baseline 1 (No training)": {"color": "#888888", "marker": "o", "ls": "--"},
    "Baseline 2 (AWGN)":        {"color": "#2C7BB6", "marker": "s", "ls": "-"},
    "Baseline 3 (LOL, RD only)": {"color": "#FDAE61", "marker": "^", "ls": "-"},
    "Ours (CBAM+Edge+TV)":      {"color": "#D7191C", "marker": "D", "ls": "-"},
}


def plot_curve(results, metric, ylabel, out_path):
    plt.figure(figsize=(7, 5))

    for label, points in results.items():
        pts = sorted(points, key=lambda p: p["bpp"])
        if not pts:
            continue
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


# ---------------------------------------------------------------- BD-rate
def bd_rate(anchor, test, metric):
    """Bjøntegaard delta rate (%). 같은 metric 품질을 내는 데 test 가 anchor 보다 비트를 몇 % 더 쓰는지.

    음수면 test 가 더 적은 비트를 쓴다. anchor/test 는 "bpp" 와 metric 키를 가진 점 리스트이고,
    곡선마다 점이 4개 이상 필요하다 (3차 다항식 피팅).
    """
    def fit(points):
        if len(points) < 4:
            raise ValueError(f"BD-rate 는 곡선마다 점이 4개 이상 필요합니다 (현재 {len(points)}개)")
        pts = sorted(points, key=lambda p: p["bpp"])
        qual = np.array([p[metric] for p in pts])
        if np.any(np.diff(qual) <= 0):
            print(f"  경고: {metric} 가 bpp 에 따라 단조 증가하지 않아 BD-rate 를 신뢰하기 어렵습니다.")
        return qual, np.polyint(np.polyfit(qual, np.log([p["bpp"] for p in pts]), 3))

    q1, i1 = fit(anchor)
    q2, i2 = fit(test)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        raise ValueError(f"두 곡선의 {metric} 구간이 겹치지 않아 BD-rate 를 계산할 수 없습니다.")
    avg_diff = ((np.polyval(i2, hi) - np.polyval(i2, lo)) - (np.polyval(i1, hi) - np.polyval(i1, lo))) / (hi - lo)
    return (np.exp(avg_diff) - 1) * 100


# ---------------------------------------------------------------- 메인
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    pairs = load_eval_pairs(CONFIG["eval_dir"])
    print(f"eval15 이미지 {len(pairs)}장 로드 완료 (device: {device})\n")

    results = {}
    print("[1/4] Baseline 1 (순정 CompressAI, 학습 없음)")
    results["Baseline 1 (No training)"] = measure_baseline1(pairs)

    print("\n[2/4] Baseline 2 (AWGN 학습)")
    results["Baseline 2 (AWGN)"] = measure("Baseline2", B2_RUNS, pairs)

    print("\n[3/4] Baseline 3 (LOL + 순수 RD Loss)")
    results["Baseline 3 (LOL, RD only)"] = measure("Baseline3", B3_RUNS, pairs)

    print("\n[4/4] Ours (CBAM + Edge + TV)")
    results["Ours (CBAM+Edge+TV)"] = measure("Ours", OURS_RUNS, pairs)

    print()
    print("[Q2 표] Ablation / 합성 저조도 학습 데이터")
    tables = {"ablation": measure("Ablation", ABLATION_RUNS, pairs),
              "synthetic": measure("Synthetic", SYNTHETIC_RUNS, pairs)}

    with open(os.path.join(CONFIG["out_dir"], "tables_q2.json"), "w") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    json_path = os.path.join(CONFIG["out_dir"], "rd_points.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n측정값 저장됨: {json_path}")

    plot_curve(results, "psnr", "PSNR (dB)",
               os.path.join(CONFIG["out_dir"], "rd_curve_psnr.png"))
    plot_curve(results, "ssim", "SSIM",
               os.path.join(CONFIG["out_dir"], "rd_curve_ssim.png"))

    print("\nBD-rate: Ours vs Baseline 3 (음수 = Ours 가 같은 품질을 더 적은 비트로 냄)")
    for metric in ("psnr", "ssim"):
        try:
            rate = bd_rate(results["Baseline 3 (LOL, RD only)"], results["Ours (CBAM+Edge+TV)"], metric)
            print(f"  {metric.upper()}: {rate:+.2f}%")
        except ValueError as e:
            print(f"  {metric.upper()}: 계산 불가 - {e}")


if __name__ == "__main__":
    # 문자열 CONFIG 키를 --키 값 으로 덮어쓸 수 있다 (예: --eval_dir /kaggle/input/.../eval15)
    parser = argparse.ArgumentParser()
    for key, value in CONFIG.items():
        if isinstance(value, str):
            parser.add_argument(f"--{key}", default=value)
    CONFIG.update(vars(parser.parse_args()))
    main()
