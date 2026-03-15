import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):

    # ImageNet stats
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ])
        for p in self.parameters():
            p.requires_grad = False

    def _preprocess(self, x):
        """Grayscale (B,C,H,W) → normalized 3-channel VGG input."""
        x = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        mean = self._MEAN.to(x.device)
        std  = self._STD.to(x.device)
        return (x - mean) / std

    def forward(self, pred, target):
        pred   = self._preprocess(pred)
        target = self._preprocess(target)

        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.mse_loss(x, y)

        return loss
