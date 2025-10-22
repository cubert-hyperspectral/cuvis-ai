import torch.nn as nn
import torch.nn.functional as F

from cuvis_ai.node.wrap import node
from cuvis_ai.utils.test import get_np_dummy_data


@node
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        # Adjust based on input image size
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        # [batch, 16, 16, 16] for 32x32 input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 64, 4, 4]

        # Flatten the tensor
        x = x.view(-1, 64 * 4 * 4)  # Adjust for input size

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


moreData = get_np_dummy_data((10, 10, 10, 10))


model = SimpleCNN(num_classes=3)

serialized = model.serialize()
# Example usage

# Print model summary
print(model)

model.fit(moreData)
