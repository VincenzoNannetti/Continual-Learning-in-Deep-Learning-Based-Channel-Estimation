import torch

checkpoint_path = r"C:\\Users\\Vincenzo_DES\\OneDrive - Imperial College London\\Year 4\\ELEC70017 - Individual Project\\Project\standard_training\\checkpoints\\unet\\unet_srcnn_trained_on_datasetMIXED\\best_model.pth"

checkpoint = torch.load(checkpoint_path, weights_only=True)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
    for key in state_dict.keys():
        print(key)


