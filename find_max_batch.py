"""
배치 사이즈 한계 테스트 전용 스크립트.
실제 학습(train.py)과 동일한 모델 구조(CBAM 포함) + 실제 LOL 데이터셋으로
forward+backward 한 번씩만 돌려서 OOM(메모리 부족)이 나는 지점을 찾는다.
저장/wandb는 안 함 (메모리 측정만 목적).

사용법 (vast.ai 인스턴스, 레포 루트에서):
    python find_max_batch.py

⚠️ 실행 전에 DATASET_PATH에 실제 LOL 데이터셋이 준비돼 있어야 함
   (hf download ... 로 받아서 unzip까지 끝낸 상태).
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
from compressai.zoo import bmshj2018_hyperprior
from models import CBAM
from dataset import LOLDataset

# ==========================================
# 실제 학습 설정과 동일하게 맞출 것 (train.py / baseline3_train.py CONFIG 참고)
# ==========================================
QUALITY = 2
CROP_SIZE = 256
CBAM_POSITION = "none"  # "encoder" / "decoder" / "none" — train.py와 동일하게 (Baseline 3: none)
DATASET_PATH = "/workspace/data/lol_dataset/our485"  # 실제 데이터셋 경로

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise SystemExit("GPU가 안 잡힙니다. nvidia-smi로 확인하세요.")

dataset = LOLDataset(root_dir=DATASET_PATH, crop_size=CROP_SIZE)
print(f"[데이터셋] {DATASET_PATH} | 이미지 {len(dataset)}장\n")


def get_real_batch(bs):
    """데이터셋에서 bs장을 랜덤으로 뽑아 실제 이미지 배치를 만든다.
    (bs가 데이터셋 크기보다 커도 되도록 복원추출)"""
    idxs = [random.randrange(len(dataset)) for _ in range(bs)]
    low_list, high_list = zip(*(dataset[i] for i in idxs))
    x = torch.stack(low_list).to(device)
    y = torch.stack(high_list).to(device)
    return x, y


def build_model():
    model = bmshj2018_hyperprior(quality=QUALITY, pretrained=True).to(device)
    if CBAM_POSITION == "encoder":
        cbam = CBAM(in_channels=model.M, reduction=16).to(device)
        layers = list(model.g_a.children())
        layers.append(cbam)
        model.g_a = nn.Sequential(*layers)
    elif CBAM_POSITION == "decoder":
        cbam = CBAM(in_channels=model.M, reduction=16).to(device)
        layers = list(model.g_s.children())
        layers.insert(0, cbam)
        model.g_s = nn.Sequential(*layers)
    model.train()
    return model


def try_batch(bs):
    """batch_size=bs로 forward+backward 한 스텝. 성공하면 peak VRAM(GB) 반환."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model()
    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = optim.Adam(params, lr=1e-4)
    aux_optimizer = optim.Adam(aux_params, lr=1e-3)

    x, y = get_real_batch(bs)  # 실제 LOL 데이터셋에서 뽑은 배치 (low, high)

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    out = model(x)
    mse = nn.functional.mse_loss(out["x_hat"], y)
    # likelihoods도 그래프에 포함시켜야 실제 학습과 동일한 메모리 패턴이 됨 (값 자체는 안 중요)
    bpp_dummy = sum(l.sum() for l in out["likelihoods"].values())
    loss = 0.0035 * (255 ** 2) * mse + 1e-9 * bpp_dummy
    loss.backward()
    optimizer.step()
    model.aux_loss().backward()
    aux_optimizer.step()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    del model, optimizer, aux_optimizer, x, y, out
    torch.cuda.empty_cache()
    return peak_gb


def find_max_batch():
    print(f"[설정] quality={QUALITY}, crop={CROP_SIZE}, cbam_position={CBAM_POSITION}")
    print(f"[GPU] {torch.cuda.get_device_name(0)} | 총 VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

    last_ok, first_fail = 0, None
    bs = 8
    # 1차: 2배씩 늘려가며 대략적인 한계 탐색
    while bs <= 1024:
        try:
            peak = try_batch(bs)
            print(f"  batch={bs:>4} -> OK  (peak {peak:.2f} GB)")
            last_ok = bs
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:>4} -> OOM")
            first_fail = bs
            break

    if first_fail is None:
        print(f"\n batch=1024까지도 문제 없음. crop_size나 quality를 올려서 다시 테스트해보세요.")
        return

    # 2차: last_ok ~ first_fail 사이 이진 탐색으로 좁히기
    lo, hi = last_ok, first_fail
    while hi - lo > 4:
        mid = (lo + hi) // 2
        # 8의 배수로 반올림 (실전에서 보기 좋은 값)
        mid = max(lo + 1, (mid // 4) * 4)
        try:
            peak = try_batch(mid)
            print(f"  [탐색] batch={mid:>4} -> OK  (peak {peak:.2f} GB)")
            lo = mid
        except torch.cuda.OutOfMemoryError:
            print(f"  [탐색] batch={mid:>4} -> OOM")
            hi = mid

    print(f"\n✅ 이 GPU에서 안전한 최대 batch_size ≈ {lo}")
    print(f"   (권장: 여유를 위해 실제 학습에는 이보다 한 단계 낮은 값을 train.py CONFIG['batch_size']에 넣으세요)")


if __name__ == "__main__":
    find_max_batch()
