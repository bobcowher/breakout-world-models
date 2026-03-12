import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

class BaseModel(nn.Module):
    
    def __init__(self):
        super().__init__()

    def save_the_model(self, filename, verbose=False):
        # Create directory if it doesn't exist
        path = "checkpoints/" + filename + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

        if verbose:
            print(f"Saved model to {path}")

    def load_the_model(self, filename, device='cuda'):
    
        path = "checkpoints/" + filename + ".pt"

        try:
            self.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded weights from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")


class Model(BaseModel):
    def __init__(self, action_dim, hidden_dim=256, observation_shape=None, obs_stack=4):
        super(Model, self).__init__()

        # CNN layers with a third layer added
        self.conv1 = nn.Conv2d(in_channels=obs_stack, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # Third convolutional layer

        # Calculate CNN output size
        conv_output_size = self.calculate_conv_output(observation_shape)
        print("conv_output_size:", conv_output_size)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self.weights_init)
    
    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))             # No pooling after second to control size
        x = F.relu(self.conv3(x))  # Pooling after third conv layer

        return x.view(-1).shape[0]

    def forward(self, x):

        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # Pooling after third conv layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers with optional dropout
        x = F.relu(self.fc1(x))
        
        output = self.output(x)

        return output

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Helper Functions
def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class WorldModel(BaseModel):

    def __init__(self):
        super().__init__()

        encoder = Encoder()

    def forward(self):

class Encoder(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64

        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2, padding=1)

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)         # (1, C_enc, H_enc, W_enc)
            self.conv_output_shape = feats.shape[1:]   # (C_enc, H_enc, W_enc)
            self.flattened_dim = feats.numel() // 1    # C_enc * H_enc * W_enc
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        self.fc_enc = nn.Linear(self.flattened_dim, embed_dim)  # ADD THIS
        # self.conv3 = nn.Conv2d()

        print(f"VAE network initialized. Input shape: {observation_shape}")

    def get_output_shape(self):
        return self.conv_output_shape

    def _conv_features(self, x):
        # Convert uint8 to float if needed (for initialization)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        return x  # (B, C_enc, H_enc, W_enc)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x
        
    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        x = self._conv_forward(x)
        x = self.fc_enc(x)
        return x
    


class Decoder(BaseModel):

    def __init__(self, observation_shape, embed_dim, conv_output_shape=[]):
        super().__init__()
        
        # Calculate the required output size dynamically
        self.conv_output_shape = (384, 4, 4)
        conv_flat_size = 384 * 4 * 4  # 4096
        
        self.fc_dec = nn.Linear(embed_dim, conv_flat_size) 

        self.deconv1 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(48, observation_shape[0], kernel_size=4, stride=2, padding=1)

    
    def _deconv_forward(self, x):
        x = x.view(-1, *self.conv_output_shape)
        x = F.elu(self.deconv1(x))
        x = F.elu(self.deconv2(x))
        x = F.elu(self.deconv3(x))
        x = F.elu(self.deconv4(x))
       
        return x

    def forward(self, x):
        x = self.fc_dec(x)
        x = self._deconv_forward(x)
        x = torch.sigmoid(x)
        return x


