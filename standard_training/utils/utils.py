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
    if model_name_lower in SUPPORTED_MODELS:
        print(f"Using pre-imported model: {model_name}")
        ModelClass = SUPPORTED_MODELS[model_name_lower]
        model = ModelClass(**model_params)
    else:
        # Try dynamic import
        try:
            # Import the model module
            module_path = f"standard_training.models.{model_name.lower()}"
            model_module = importlib.import_module(module_path)
            
            # Get model class (assume it's capitalized)
            model_class_name = model_name.upper()
            ModelClass = getattr(model_module, model_class_name)
            
            # Instantiate the model with parameters
            model = ModelClass(**model_params)
        except ImportError as e:
            raise ImportError(f"Could not import model module for '{model_name}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Could not find model class '{model_class_name}' in module: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")
    
    # Load pretrained weights if specified
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
