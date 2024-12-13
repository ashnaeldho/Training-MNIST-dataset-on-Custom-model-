import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # Small convolutional network with required components
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),  # Batch normalization
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.25),  # Dropout
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create and save the model
model = TestModel()
torch.save(model, 'best_model.pth') 