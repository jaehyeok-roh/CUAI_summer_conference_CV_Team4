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

class SmoothedEdgeLoss(nn.Module):
    def __init__(self, blur_ksize=7, blur_sigma=1.5, use_gray=True, eps=1e-6):
        super().__init__()
        kx, ky = make_sobel()
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)
        self.register_buffer('gauss', gaussian_kernel(blur_ksize, blur_sigma))
        self.use_gray = use_gray
        self.eps = eps

    def forward(self, pred, target):
        e_pred = edge_magnitude(pred, self.sobel_x, self.sobel_y, self.eps, self.use_gray)
        e_tgt  = edge_magnitude(target, self.sobel_x, self.sobel_y, self.eps, self.use_gray)
        return F.l1_loss(blur(e_pred, self.gauss), blur(e_tgt, self.gauss))

class RateDistortionEdgeLoss(nn.Module):
    def __init__(self, lmbda=0.0035, edge_weight=0.1, blur_ksize=7, blur_sigma=1.5, use_gray=True, mse_blur_sigma=1.0, mse_blur_ksize=None):
        super().__init__()
        self.lmbda = lmbda
        self.edge_weight = edge_weight
        self.mse = nn.MSELoss()
        self.edge = SmoothedEdgeLoss(blur_ksize, blur_sigma, use_gray)
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

        # BPP 계산 시 정보이론에 맞게 log2 사용 및 보호
        bpp = sum((torch.log2(l.clamp(min=1e-9)).sum() / (-num_pixels)) for l in out_net['likelihoods'].values())

        if self.mse_gauss is not None:
            mse_val = self.mse(blur(x_hat, self.mse_gauss), blur(target, self.mse_gauss))
        else:
            mse_val = self.mse(x_hat, target)

        edge_val = self.edge(x_hat, target)

        distortion_term = self.lmbda * (255 ** 2) * mse_val
        edge_term = self.edge_weight * edge_val

        loss = bpp + distortion_term + edge_term
        return loss, {'bpp': bpp.item(), 'distortion_term': distortion_term.item(), 'edge_term': edge_term.item()}