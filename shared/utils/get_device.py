import torch

def get_device(config):
    """Get the appropriate device based on config."""
    device_name = config.get('hardware', {}).get('device', 'auto').lower()
    if device_name == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif device_name == 'cpu':
        device = torch.device("cpu")
    else:
        print(f"Warning: Invalid device name '{device_name}'. Using auto-detection.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device