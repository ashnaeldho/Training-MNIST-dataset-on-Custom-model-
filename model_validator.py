import torch
import torch.nn as nn
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_batch_norm(model):
    return any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) 
              for module in model.modules())

def check_dropout(model):
    return any(isinstance(module, nn.Dropout) for module in model.modules())

def check_global_avg_pooling(model):
    return any(isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d))
              for module in model.modules())

def validate_model():
    try:
        # Load the model
        model = torch.load('best_model.pth', map_location=torch.device('cpu'))
        
        # Check total parameters
        param_count = count_parameters(model)
        if param_count > 20000:
            print(f"❌ Model has {param_count} parameters, which exceeds the limit of 20,000")
            return False
        else:
            print(f"✅ Model has {param_count} parameters")

        # Check batch normalization
        if not check_batch_norm(model):
            print("❌ Model does not use batch normalization")
            return False
        print("✅ Model uses batch normalization")

        # Check dropout
        if not check_dropout(model):
            print("❌ Model does not use dropout")
            return False
        print("✅ Model uses dropout")

        # Check global average pooling
        if not check_global_avg_pooling(model):
            print("❌ Model does not use global average pooling")
            return False
        print("✅ Model uses global average pooling")

        return True

    except Exception as e:
        print(f"Error during model validation: {str(e)}")
        return False

if __name__ == "__main__":
    success = validate_model()
    sys.exit(0 if success else 1) 