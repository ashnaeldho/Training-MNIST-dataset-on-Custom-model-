import torch
import torch.nn as nn
import sys
import os

# Define the same model class here for loading
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(32, 16, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv7 = nn.Conv2d(16, 10, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.dropout3(x)
        
        x = self.conv7(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

def count_parameters(model):
    # Count only trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("::group::Parameter Count Details")
    print("\nDetailed Parameter Breakdown:")
    print("--------------------------------")
    
    # Print every layer's parameters with more detail
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            print(f"{name}: {type(module).__name__}: {trainable:,} trainable parameters")
            
            # Print detailed parameter shapes for this module
            for param_name, param in module.named_parameters():
                trainable_str = "trainable" if param.requires_grad else "non-trainable"
                print(f"  └─ {param_name}: shape {list(param.shape)} = {param.numel():,} parameters ({trainable_str})")
    
    print("\nTotal Parameter Count:")
    print("--------------------------------")
    print(f"Total trainable parameters: {trainable_params:,}")  # Should be 18,314
    
    # Also show total parameters including non-trainable
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (including non-trainable): {total_params:,}")
    print("--------------------------------")
    print("::endgroup::")
    
    return trainable_params

def check_batch_norm(model):
    return any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) 
              for module in model.modules())

def check_dropout(model):
    return any(isinstance(module, nn.Dropout) for module in model.modules())

def check_global_avg_pooling(model):
    return any(isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d))
              for module in model.modules())

def print_github_output(message, is_error=False):
    """Helper function to format output for GitHub Actions."""
    if is_error:
        print(f"::error::❌ {message}")
    else:
        print(f"::notice::✅ {message}")

def validate_model():
    print("::group::Model Validation Results")
    try:
        model_path = 'best_model.pth'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print_github_output(f"Model file '{model_path}' not found in current directory: {os.getcwd()}", True)
            print("Contents of current directory:")
            for file in os.listdir():
                print(f"  - {file}")
            return False
            
        # Load the model
        try:
            model = TestModel()
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            
            # Print model architecture
            print("::group::Model Architecture")
            print(model)
            print("::endgroup::")
            
        except Exception as load_error:
            print_github_output(f"Failed to load model: {str(load_error)}", True)
            return False

        # Check total parameters
        param_count = count_parameters(model)
        if param_count > 20000:
            print_github_output(f"Model has {param_count:,} parameters (exceeds limit of 20,000)", True)
            return False
        else:
            print_github_output(f"Model has {param_count:,} parameters (under 20,000 limit)")

        # Check batch normalization
        if not check_batch_norm(model):
            print_github_output("Model does not use batch normalization", True)
            return False
        print_github_output("Model uses batch normalization")

        # Check dropout
        if not check_dropout(model):
            print_github_output("Model does not use dropout", True)
            return False
        print_github_output("Model uses dropout")

        # Check global average pooling
        if not check_global_avg_pooling(model):
            print_github_output("Model does not use global average pooling", True)
            return False
        print_github_output("Model uses global average pooling")

        return True

    except Exception as e:
        print_github_output(f"Error during model validation: {str(e)}", True)
        return False
    finally:
        print("::endgroup::")

if __name__ == "__main__":
    success = validate_model()
    sys.exit(0 if success else 1) 