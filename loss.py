import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_gray(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

def make_sobel():
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    ky = kx.t().contiguous()
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)

def edge_magnitude(x, sobel_x, sobel_y, eps=1e-6, use_gray=True):
    if use_gray: x = rgb_to_gray(x)
    C = x.shape[1]
    sx = sobel_x.to(x.device).repeat(C, 1, 1, 1)
    sy = sobel_y.to(x.device).repeat(C, 1, 1, 1)
    gx = F.conv2d(x, sx, padding=1, groups=C)
    gy = F.conv2d(x, sy, padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + eps)

def gaussian_kernel(ksize=5, sigma=1.0):
    ax = torch.arange(ksize) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).view(1, 1, ksize, ksize)

def blur(x, gk):
    C = x.shape[1]
    g = gk.to(x.device).repeat(C, 1, 1, 1)
    pad = gk.shape[-1] // 2
    return F.conv2d(x, g, padding=pad, groups=C)

# --- Total Variation Loss 모듈 ---
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 인접한 상하/좌우 픽셀 간의 차이 절대값 평균 (Checkerboard 억제 핵심)
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w

# --- [수정됨] SmoothedEdgeLoss -> SharpEdgeLoss ---
class SharpEdgeLoss(nn.Module):
    def __init__(self, use_gray=True, eps=1e-6):
        super().__init__()
        kx, ky = make_sobel()
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)
        self.use_gray = use_gray
        self.eps = eps

    def forward(self, pred, target):
        # 엣지 크기(Magnitude) 추출
        e_pred = edge_magnitude(pred, self.sobel_x, self.sobel_y, self.eps, self.use_gray)
        e_tgt  = edge_magnitude(target, self.sobel_x, self.sobel_y, self.eps, self.use_gray)
        
        # 가우시안 블러를 완전히 제거하고 픽셀 단위로 직접 L1 Loss 계산
        return F.l1_loss(e_pred, e_tgt)

# --- [수정됨] RateDistortionEdgeLoss 반영 ---
class RateDistortionEdgeLoss(nn.Module):
    # edge 블러 관련 인자(blur_ksize, blur_sigma) 제거
    def __init__(self, lmbda=0.0035, edge_weight=10.0, tv_weight=40.0, use_gray=True, mse_blur_sigma=0.0, mse_blur_ksize=None):
        super().__init__()
        self.lmbda = lmbda
        self.edge_weight = edge_weight
        self.tv_weight = tv_weight
        self.mse = nn.MSELoss()
        
        # 교체된 SharpEdgeLoss 적용
        self.edge = SharpEdgeLoss(use_gray)  
        self.tv = TotalVariationLoss()  
        
        self.mse_blur_sigma = float(mse_blur_sigma)
        
        if self.mse_blur_sigma > 0:
            if mse_blur_ksize is None:
                mse_blur_ksize = int(2 * math.ceil(3 * self.mse_blur_sigma) + 1)
            self.register_buffer('mse_gauss', gaussian_kernel(int(mse_blur_ksize), self.mse_blur_sigma))
        else:
            self.register_buffer('mse_gauss', None)

    def forward(self, out_net, target):
        x_hat = out_net['x_hat']
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # BPP 계산
        bpp = sum((torch.log2(l.clamp(min=1e-9)).sum() / (-num_pixels)) for l in out_net['likelihoods'].values())

        if self.mse_gauss is not None:
            mse_val = self.mse(blur(x_hat, self.mse_gauss), blur(target, self.mse_gauss))
        else:
            mse_val = self.mse(x_hat, target)

        edge_val = self.edge(x_hat, target)
        tv_val = self.tv(x_hat)  

        distortion_term = self.lmbda * (255 ** 2) * mse_val
        edge_term = self.edge_weight * edge_val
        tv_term = self.tv_weight * tv_val  

        loss = bpp + distortion_term + edge_term + tv_term
        return loss, {
            'bpp': bpp.item(), 
            'distortion_term': distortion_term.item(), 
            'edge_term': edge_term.item(),
            'tv_term': tv_term.item()
        }