"""
Filename: standard_training/utils/utils.py
Author: Vincenzo Nannetti
Date: 04/03/2025
Description: Utility functions for model and data loading

Dependencies:
    - PyTorch
    - yaml
"""
import torch
import importlib
import os

from standard_training.models.dncnn import DnCNN
from standard_training.models.srcnn import SRCNN
from standard_training.models.unet import UNetModel
from standard_training.models.denoising_autoencoder import DenoisingAutoencoder, DenoisingResAutoencoder
from standard_training.models.combined_models.ae_srcnn import CombinedModel_AESRCNN 
from standard_training.models.combined_models.srcnn_dncnn import CombinedModel_SRCNNDnCNN
from standard_training.models.combined_models.unet_srcnn import UNetCombinedModel 

SUPPORTED_MODELS = {
    "dncnn": DnCNN,
    "srcnn": SRCNN,
    "unet": UNetModel,
    "combined_srcnn_dncnn": CombinedModel_SRCNNDnCNN,
    "combined_unet_srcnn": UNetCombinedModel,
    "denoising_autoencoder": DenoisingAutoencoder,
    "denoising_res_autoencoder": DenoisingResAutoencoder,
    "combined_ae_srcnn": CombinedModel_AESRCNN,
}

def load_model(config):
    """Load the model based on the configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        model: PyTorch model instance.
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name', '')
    model_params = model_config.get('params', {})
    
    if not model_name:
        raise ValueError("Model name not specified in config.")
    
    # Check if model is in our predefined list
    model_name_lower = model_name.lower()
    ModelClass = SUPPORTED_MODELS.get(model_name_lower)

    if ModelClass is None:
        raise ValueError(f"Model '{model_name}' not supported. Supported models: {', '.join(SUPPORTED_MODELS.keys())}")

    # Instantiate the model
    try:
        if model_name_lower == "combined_unet_srcnn":
            unet_combined_args = {
                'unet_args': model_params,  # model_params is config.model.params
                'pretrained_unet': model_config.get('pretrained_unet', None),
                'pretrained_srcnn': model_config.get('pretrained_srcnn', None)
            }
            print(f"Instantiating {model_name} with specific args for UNetCombinedModel")
            model = ModelClass(**unet_combined_args)
        elif ModelClass: # Check if ModelClass was successfully obtained (either from SUPPORTED_MODELS or dynamic import)
            print(f"Using model: {model_name}")
            model = ModelClass(**model_params)
        else:
            # This case should ideally be caught earlier if dynamic import also fails
            raise RuntimeError(f"Model class for '{model_name}' could not be determined.")
            
    except TypeError as e:
        raise TypeError(f"Error instantiating model '{model_name}' with params {model_params if model_name_lower != 'combined_unet_srcnn' else unet_combined_args}: {e}")
    except Exception as e:
        raise RuntimeError(f"General error during model instantiation for '{model_name}': {e}")
    
    # Load pretrained weights if specified for the entire model
    # This is separate from component-specific pretraining (e.g. pretrained_unet for UNetCombinedModel)
    # which should be handled during the model's own __init__.
    pretrained_path = model_config.get('pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            # Load from training checkpoint
            model.load_state_dict(checkpoint['model_state_dict'], 
                                 strict=model_config.get('strict_load', True))
        else:
            # Direct state dict
            model.load_state_dict(checkpoint, 
                                 strict=model_config.get('strict_load', True))
            
    return model

def load_data(config, mode='train'):
    """Load data according to the configuration.
    
    Args:
        config (dict): Configuration dictionary.
        mode (str): Mode of operation ('train', 'eval').
        
    Returns:
        tuple: (dataloaders, normalisation_info)
    """
    from standard_training.datasets.dataset_utils import load_data as load_data_impl
    return load_data_impl(config, mode)
