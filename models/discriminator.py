import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class Discriminator(BaseModel):
    """
    Discriminator network for adversarial training of encoder/decoder.

    Takes an image and outputs a single logit indicating real (1) or fake (0).
    """

    def __init__(self, input_shape=(3, 128, 128)):
        super().__init__()

        channels, height, width = input_shape

        # Convolutional layers with LeakyReLU (standard for discriminators)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Calculate final spatial size: 128 -> 64 -> 32 -> 16 -> 8
        final_size = height // 16
        self.fc = nn.Linear(512 * final_size * final_size, 1)

        print(f"Discriminator initialized for input shape: {input_shape}")
        print(f"Final conv output: 512 x {final_size} x {final_size}")

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor in [0, 1]

        Returns:
            logits: (B, 1) real/fake logits
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        return logits
