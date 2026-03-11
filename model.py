import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module):
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

    def save_the_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)

    def load_the_model(self, filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")

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
