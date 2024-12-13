import torch
import torch.nn as nn
import ast

# Load the model file
model_file = 'best_model.pth'

# Load the model architecture
class Net(nn.Module):
    # Your model definition here
    # For example, you can copy the model definition from Main.py
    def __init__(self):
        super(Net, self).__init__()
        # ... (model layers)

    def forward(self, x):
        # ... (forward pass)
        return x

# Load the model
model = Net()
model.load_state_dict(torch.load(model_file))

# Check total parameter count
total_params = sum(p.numel() for p in model.parameters())
if total_params >= 20000:
    raise ValueError(f"Total parameter count is {total_params}, which exceeds 20,000.")

# Check for Batch Normalization
has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
if not has_batch_norm:
    raise ValueError("Batch Normalization is not used in the model.")

# Check for Dropout
has_dropout = any(isinstance(layer, nn.Dropout) for layer in model.modules())
if not has_dropout:
    raise ValueError("Dropout is not used in the model.")

# Check for Global Average Pooling
has_global_avg_pooling = any(isinstance(layer, nn.AdaptiveAvgPool2d) for layer in model.modules())
if not has_global_avg_pooling:
    raise ValueError("Global Average Pooling is not used in the model.")

print("All checks passed successfully.")