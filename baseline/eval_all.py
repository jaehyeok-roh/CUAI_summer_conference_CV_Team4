"""
논문 표용 통합 평가: 모든 방법을 같은 평가 경로로 LOL-v1 eval15 에서 한 번에 잰다.

평가 경로: 코덱 복원 영상과 향상 모델 출력을 모두 8비트로 양자화한다 (수신 측이 실제로 다루는 영상과 같게).
bpp 는 likelihood 기반이고 PSNR/SSIM 함수는 rd_curve.py 와 같다. 비교 지표는 log-bpp 선형 보간 BD 와 부호 검정이다.

  Standard codec / Compress -> Enhance / Enhance -> Compress / Standard codec + refiner : 품질 1~8
  Enhance only                                                                       : 압축 없음 (상한)
  Joint codec (RD only) / Joint codec (CUAI v1) / Joint codec + refiner              : 품질 2,4,6,8 (W&B 체크포인트)
  Control (codec frozen) / v2 (noise)                                                : 품질 1,3,5,7
  Ours: v2 (STE)                                                                     : 품질 1,3,5

--ckpt_root 아래에서 refiner_Standard_codec.pth, refiner_Ours.pth, frozen_q*.pth, joint_q*.pth, joint_ste_q*.pth 를 이름으로 찾는다.
W&B 체크포인트는 WANDB_API_KEY 환경 변수로 받는다. 체크포인트를 찾지 못한 방법은 건너뛰고 나머지를 평가한다.

실행 (two_stage.py 와 같은 준비 후, 레포 루트에서):
    python baseline/eval_all.py --eval_dir <LOL 경로>/eval15 --weights pretrained_weights/LOL_v1.pth --ckpt_root /kaggle/input
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from compressai.zoo import bmshj2018_hyperprior

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/
from models import HyperpriorWithCBAM
from rd_curve import B3_RUNS, OURS_RUNS, compute_psnr, compute_ssim, device, fetch_from_wandb, load_eval_pairs, load_state_dict_into, pad_to_64
from two_stage import build_retinexformer, load_retinexformer

STD, C2E, E2C, STD_REF = "Standard codec", "Compress -> Enhance", "Enhance -> Compress", "Standard codec + refiner"
B3, CUAI, CUAI_REF = "Joint codec (RD only)", "Joint codec (CUAI v1)", "Joint codec + refiner"
CONTROL, NOISE, OURS = "Control (codec frozen)", "v2 (noise)", "Ours: v2 (STE)"
COMPARISONS = [  # (test, anchor). 앞의 세 개는 이미지별 부호 검정도 한다
    (OURS, CONTROL), (NOISE, CONTROL), (OURS, NOISE),
    (STD_REF, C2E), (CUAI_REF, STD_REF), (CUAI, B3), (STD_REF, E2C), (OURS, E2C),
]


def q8(x):
    """0~1 영상을 8비트로 양자화 (float 유지)"""
    return (x.clamp(0, 1) * 255).round() / 255


@torch.no_grad()
def compress(codec, img):
    """3xHxW -> (8비트 복원 영상, bpp)"""
    x, h, w = pad_to_64(img)
    out = codec(x.unsqueeze(0).to(device))
    bpp = sum(torch.log2(l.clamp(min=1e-9)).sum() / (-h * w) for l in out["likelihoods"].values()).item()
    return q8(out["x_hat"][0, :, :h, :w]), bpp


@torch.no_grad()
def enhance(model, img):
    x, h, w = pad_to_64(img)
    return q8(model(x.unsqueeze(0).to(device))[0, :, :h, :w])


def codec_then(codec, enhancer=None):
    """저조도 영상 -> 코덱 -> (향상 모델) 파이프라인"""
    def run(low):
        y, bpp = compress(codec, low)
        return (y if enhancer is None else enhance(enhancer, y)), bpp
    return run


def build_methods(weights, ckpt_root):
    """이름 -> [(품질, 파이프라인을 만드는 함수)]. 모델은 평가할 때 품질마다 만든다."""
    def find(name):
        found = glob.glob(os.path.join(ckpt_root, "**", name), recursive=True)
        if not found:
            raise FileNotFoundError(name)
        return found[0]

    def std_codec(q):
        return bmshj2018_hyperprior(quality=q, pretrained=True).eval().to(device)

    def joint_codec(runs, q):
        quality, run_name, filename, cbam, match = runs[q]
        ckpt = fetch_from_wandb(run_name, filename, match)
        if ckpt is None:
            raise FileNotFoundError(f"W&B {run_name}")
        model = HyperpriorWithCBAM(quality=quality, cbam_position=cbam, pretrained=False)
        load_state_dict_into(model, ckpt)
        return model.eval().to(device)

    def trained(tag, q):
        state = torch.load(find(f"{tag}_q{q}.pth"), map_location="cpu")
        codec = bmshj2018_hyperprior(quality=q, pretrained=False)
        codec.load_state_dict(state["codec"])
        refiner = build_retinexformer()
        refiner.load_state_dict(state["refiner"])
        return codec_then(codec.eval().to(device), refiner.eval().to(device))

    pre = load_retinexformer(weights)
    refiners = {}  # 여러 품질이 같은 refiner 를 쓴다

    def refiner(name):
        if name not in refiners:
            refiners[name] = load_retinexformer(find(name))
        return refiners[name]

    def e2c(q):
        codec = std_codec(q)
        return lambda low: compress(codec, enhance(pre, low))

    all_q, even_q = range(1, 9), (2, 4, 6, 8)
    return {
        STD: [(q, lambda q=q: codec_then(std_codec(q))) for q in all_q],
        C2E: [(q, lambda q=q: codec_then(std_codec(q), pre)) for q in all_q],
        E2C: [(q, lambda q=q: e2c(q)) for q in all_q],
        "Enhance only": [(0, lambda: lambda low: (enhance(pre, low), 0.0))],
        STD_REF: [(q, lambda q=q: codec_then(std_codec(q), refiner("refiner_Standard_codec.pth"))) for q in all_q],
        B3: [(q, lambda q=q: codec_then(joint_codec(B3_RUNS, q))) for q in even_q],
        CUAI: [(q, lambda q=q: codec_then(joint_codec(OURS_RUNS, q))) for q in even_q],
        CUAI_REF: [(q, lambda q=q: codec_then(joint_codec(OURS_RUNS, q), refiner("refiner_Ours.pth"))) for q in even_q],
        CONTROL: [(q, lambda q=q: trained("frozen", q)) for q in (1, 3, 5, 7)],
        NOISE: [(q, lambda q=q: trained("joint", q)) for q in (1, 3, 5, 7)],
        OURS: [(q, lambda q=q: trained("joint_ste", q)) for q in (1, 3, 5)],
    }


def evaluate(methods, pairs):
    """이름 -> {품질: [[이미지, bpp, PSNR, SSIM], ...]}. 모델을 만들지 못한 점은 건너뛴다."""
    results = {}
    for name, points in methods.items():
        for q, make in points:
            try:
                run = make()
            except FileNotFoundError as e:
                print(f"  건너뜀: {name} q={q} ({e} 없음)", flush=True)
                continue
            rows = []
            for low, high, image in pairs:
                y, bpp = run(low)
                y, gt = y.unsqueeze(0), high.unsqueeze(0).to(device)
                rows.append([image, bpp, compute_psnr(y, gt, data_range=1.0).item(), compute_ssim(y, gt, data_range=1.0).item()])
            results.setdefault(name, {})[q] = rows
            m = np.mean([r[1:] for r in rows], axis=0)
            print(f"{name} q={q}: bpp {m[0]:.4f} / PSNR {m[1]:.2f} / SSIM {m[2]:.4f}", flush=True)
            del run
            if device == "cuda":
                torch.cuda.empty_cache()
    return results


def mean_curve(points):
    return sorted(tuple(float(np.mean([r[1 + m] for r in rows])) for m in range(3)) for rows in points.values())


def image_curve(points, i):
    return sorted(tuple(rows[i][1:]) for rows in points.values())


def bd_quality(anchor, test, k):
    """겹치는 log-bpp 구간에서 test - anchor 품질(k: 1 PSNR, 2 SSIM) 차이의 평균. 겹치지 않으면 None"""
    ra, rt = np.log([p[0] for p in anchor]), np.log([p[0] for p in test])
    lo, hi = max(ra[0], rt[0]), min(ra[-1], rt[-1])
    if lo >= hi:
        return None
    x = np.linspace(lo, hi, 200)
    return float(np.mean(np.interp(x, rt, [p[k] for p in test]) - np.interp(x, ra, [p[k] for p in anchor])))


def bd_rate(anchor, test, k):
    """겹치는 품질 구간에서 같은 품질에 드는 비트의 평균 비율 차이 (음수 = test 가 비트를 아낌). 품질이 단조 증가하지 않으면 None"""
    qa, qt = np.array([p[k] for p in anchor]), np.array([p[k] for p in test])
    lo, hi = max(qa.min(), qt.min()), min(qa.max(), qt.max())
    if lo >= hi or np.any(np.diff(qa) <= 0) or np.any(np.diff(qt) <= 0):
        return None
    x = np.linspace(lo, hi, 200)
    return float(np.exp(np.mean(np.interp(x, qt, np.log([p[0] for p in test])) - np.interp(x, qa, np.log([p[0] for p in anchor])))) - 1)


def sign_test(wins, n):
    """양측 부호 검정 p 값"""
    tail = sum(math.comb(n, i) for i in range(max(wins, n - wins), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(results):
    """평균 곡선과 비교별 BD 를 찍는다. 반환: {(test, anchor): 지표}"""
    def show(v, pct=False):
        return "-" if v is None else (f"{v * 100:+.1f}%" if pct else f"{v:+.4f}")

    for name, points in results.items():
        print(f"[curve] {name}: " + " | ".join(f"{b:.3f}bpp {p:.2f}dB {s:.4f}" for b, p, s in mean_curve(points)))
    summary = {}
    for i, (test, anchor) in enumerate(COMPARISONS):
        if test not in results or anchor not in results:
            continue
        a, t = mean_curve(results[anchor]), mean_curve(results[test])
        entry = {"bd_psnr": bd_quality(a, t, 1), "bd_ssim": bd_quality(a, t, 2), "bd_rate_psnr": bd_rate(a, t, 1), "bd_rate_ssim": bd_rate(a, t, 2)}
        print(f"[BD] {test} vs {anchor}: BD-PSNR {show(entry['bd_psnr'])}, BD-SSIM {show(entry['bd_ssim'])}, "
              f"BD-rate(PSNR) {show(entry['bd_rate_psnr'], True)}, BD-rate(SSIM) {show(entry['bd_rate_ssim'], True)}")
        if i < 3:
            n_images = len(next(iter(results[anchor].values())))
            for k, metric in ((1, "PSNR"), (2, "SSIM")):
                v = [x for x in (bd_quality(image_curve(results[anchor], j), image_curve(results[test], j), k) for j in range(n_images)) if x is not None]
                wins = sum(x > 0 for x in v)
                entry[f"wins_{metric}"] = (wins, len(v))
                print(f"[image] {test} vs {anchor} BD-{metric}: {wins}/{len(v)} 장 이김 (평균 {np.mean(v):+.4f}, p={sign_test(wins, len(v)):.4f})")
        summary[(test, anchor)] = entry
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="./LOL_Dataset/lol_dataset/eval15")
    parser.add_argument("--weights", default="./pretrained_weights/LOL_v1.pth")
    parser.add_argument("--ckpt_root", default="/kaggle/input")
    parser.add_argument("--out", default="./results/eval_all.json")
    args = parser.parse_args()

    results = evaluate(build_methods(args.weights, args.ckpt_root), load_eval_pairs(args.eval_dir))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    report(results)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
