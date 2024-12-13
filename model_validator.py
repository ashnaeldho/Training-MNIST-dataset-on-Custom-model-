import torch
import torch.nn as nn
import sys
import os

# Define the same model class here for loading
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    
    print("::group::Parameter Count Details")
    print("\nDetailed Parameter Breakdown:")
    print("--------------------------------")
    
    # Print every layer's parameters
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {type(module).__name__}: {params:,} parameters")
            
            # Print detailed parameter shapes for this module
            for param_name, param in module.named_parameters():
                print(f"  └─ {param_name}: shape {list(param.shape)} = {param.numel():,} parameters")
    
    print("\nTotal Parameter Count:")
    print("--------------------------------")
    print(f"Total trainable parameters: {total_params:,}")  # Should be 18,314
    print("--------------------------------")
    print("::endgroup::")
    
    return total_params

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