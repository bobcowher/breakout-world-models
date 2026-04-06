import torch
import torch.nn.functional as F

def gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):
    """Create a Gaussian kernel for SSIM."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Args:
        img1: (B, C, H, W) tensor in [0, 1]
        img2: (B, C, H, W) tensor in [0, 1]
        window_size: Size of Gaussian window
        size_average: If True, return mean SSIM; else return per-pixel

    Returns:
        SSIM score in [0, 1] where 1 = identical
    """
    channels = img1.size(1)

    # Create Gaussian window
    window = gaussian_kernel(window_size, sigma=1.5, channels=channels).to(img1.device)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def ssim_loss(img1, img2, window_size=11):
    """
    SSIM loss (1 - SSIM) for optimization.

    Returns value in [0, 1] where 0 = identical images.
    """
    return 1 - ssim(img1, img2, window_size)
