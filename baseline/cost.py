"""
파이프라인별 계산량: 엣지(송신) 쪽과 수신 쪽의 파라미터 수, FLOPs, CPU 지연시간.

  - Standard codec      : 엣지 = 코덱 인코더,                 수신 = 코덱 디코더
  - Enhance -> Compress : 엣지 = Retinexformer + 코덱 인코더, 수신 = 코덱 디코더
  - Compress -> Enhance : 엣지 = 코덱 인코더,                 수신 = 코덱 디코더 + Retinexformer
  - Ours                : 엣지 = 코덱 인코더,                 수신 = 코덱 디코더 + CBAM
  - Ours + refiner      : 엣지 = 코덱 인코더,                 수신 = 코덱 디코더 + CBAM + Retinexformer

코덱 인코더는 g_a, h_a, h_s (compress() 가 scale 계산에 h_s 를 쓴다), 디코더는 h_s, g_s 다.
엔트로피 코딩(range coder)은 같은 코덱이면 모든 파이프라인에서 같으므로 뺐다.
계산량은 가중치 값과 무관해서 무작위 초기화 모델로 잰다. 입력은 LOL 해상도(600x400).

준비: two_stage.py 와 같이 baseline/RetinexFormer_arch.py 가 있어야 한다.
실행:
    python baseline/cost.py --qualities 2 6
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # baseline/
from models import HyperpriorWithCBAM
from two_stage import build_retinexformer


def params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


@torch.no_grad()
def cost(fn, runs):
    """fn() 의 FLOPs 와 CPU 지연시간(ms, runs 번 중앙값). FLOPs 를 세는 첫 실행이 워밍업을 겸한다."""
    with FlopCounterMode(display=False) as counter:
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    # FlopCounterMode 는 곱셈 1회를 2 로 센다. 논문들이 쓰는 FLOPs(fvcore/thop 의 MACs 관행)에 맞춰 반으로 나눈다
    return counter.get_total_flops() // 2, statistics.median(times)


@torch.no_grad()
def pipelines(quality, enhancer, h=400, w=600, runs=5):
    ours = HyperpriorWithCBAM(quality=quality, cbam_position="decoder", pretrained=False).eval()
    codec = ours.base_model  # Standard codec 은 같은 구조에서 CBAM 만 빼고 잰다
    x = torch.rand(1, 3, h, w)
    x64 = F.pad(x, (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64), mode="reflect")
    y_hat = torch.round(codec.g_a(x64))
    z_hat = torch.round(codec.h_a(torch.abs(y_hat)))

    def encode():
        codec.h_s(torch.round(codec.h_a(torch.abs(codec.g_a(x64)))))

    def decode(cbam):
        codec.h_s(z_hat)
        codec.g_s(y_hat + ours.cbam_alpha * ours.cbam(y_hat) if cbam else y_hat)

    enc = (params(codec.g_a, codec.h_a, codec.h_s, codec.entropy_bottleneck), cost(encode, runs))
    dec = (params(codec.h_s, codec.g_s, codec.entropy_bottleneck), cost(lambda: decode(False), runs))
    dec_cbam = (dec[0] + params(ours.cbam) + ours.cbam_alpha.numel(), cost(lambda: decode(True), runs))
    enh = (params(enhancer), cost(lambda: enhancer(x), runs))  # 향상 전후 해상도가 같아 한 번만 잰다

    def side(*parts):
        return {"params": sum(p for p, _ in parts), "flops": sum(c[0] for _, c in parts), "ms": sum(c[1] for _, c in parts)}

    return {
        "Standard codec": {"edge": side(enc), "receiver": side(dec)},
        "Enhance -> Compress": {"edge": side(enh, enc), "receiver": side(dec)},
        "Compress -> Enhance": {"edge": side(enc), "receiver": side(dec, enh)},
        "Ours": {"edge": side(enc), "receiver": side(dec_cbam)},
        "Ours + refiner": {"edge": side(enc), "receiver": side(dec_cbam, enh)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualities", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="./results/rd_curve/cost.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    enhancer = build_retinexformer().eval()

    results = {}
    for quality in args.qualities:
        results[quality] = pipelines(quality, enhancer, runs=args.runs)
        print(f"\n[quality {quality}] 600x400, CPU {torch.get_num_threads()} threads, 엔트로피 코딩 제외")
        print(f"  {'':<21}{'엣지 params':>12}{'GFLOPs':>8}{'ms':>7}{'수신 params':>12}{'GFLOPs':>8}{'ms':>7}")
        for name, row in results[quality].items():
            e, r = row["edge"], row["receiver"]
            print(f"  {name:<21}{e['params'] / 1e6:>11.2f}M{e['flops'] / 1e9:>8.1f}{e['ms']:>7.0f}"
                  f"{r['params'] / 1e6:>11.2f}M{r['flops'] / 1e9:>8.1f}{r['ms']:>7.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"저장됨: {args.out}")


if __name__ == "__main__":
    main()
