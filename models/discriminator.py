import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class SimilarityDiscriminator(BaseModel):
    """
    Similarity-based discriminator for reconstruction quality.

    Instead of classifying "real vs fake", learns to predict whether two images
    are the same or different. This provides more stable training and better
    gradients for spatial accuracy (e.g., paddle position).

    Takes two images and outputs similarity score (1 = same, 0 = different).
    """

    def __init__(self, input_shape=(3, 128, 128), embed_dim=512):
        super().__init__()

        channels, height, width = input_shape
        self.embed_dim = embed_dim

        # Shared encoder for both images (Siamese architecture)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Calculate final spatial size: 128 -> 64 -> 32 -> 16 -> 8
        final_size = height // 16
        self.flatten_dim = 512 * final_size * final_size

        # Project to embedding
        self.fc_embed = nn.Linear(self.flatten_dim, embed_dim)

        # Comparison head - takes concatenated embeddings
        self.fc_compare = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        print(f"SimilarityDiscriminator initialized for input shape: {input_shape}")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Comparison head: 2x{embed_dim} -> 256 -> 1")

    def encode(self, x):
        """
        Encode an image to an embedding vector.

        Args:
            x: (B, C, H, W) image tensor in [0, 1]

        Returns:
            embed: (B, embed_dim) embedding vector
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        x = x.view(x.size(0), -1)
        embed = self.fc_embed(x)

        return embed

    def forward(self, img1, img2):
        """
        Compare two images for similarity.

        Args:
            img1: (B, C, H, W) first image
            img2: (B, C, H, W) second image

        Returns:
            similarity_logits: (B, 1) logits for similarity (1 = same, 0 = different)
        """
        # Encode both images using shared encoder (Siamese)
        embed1 = self.encode(img1)
        embed2 = self.encode(img2)

        # Concatenate embeddings and compare
        combined = torch.cat([embed1, embed2], dim=1)
        similarity_logits = self.fc_compare(combined)

        return similarity_logits


# Keep old Discriminator for backwards compatibility
class Discriminator(BaseModel):
    """
    DEPRECATED: Original adversarial discriminator.
    Use SimilarityDiscriminator instead for better stability.
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
