"""
Online model manager for loading trained offline models and handling dynamic task switching.
Manages LoRA adapters, batch normalization statistics, and replay buffers per domain.
"""

import torch
import os
from typing import Dict, Any, Optional
import time
from main_algorithm_v2.offline.replay_buffer import ReplayBuffer
from .ewc import OnlineFisherInformationManager, OnlineEWCLoss, create_online_ewc_manager

class OnlineModelManager:
    """
    Manages the trained offline model for online inference with dynamic task switching.
    """
    
    def __init__(self, checkpoint_path: str, device: torch.device, 
                 enable_ewc: bool = False, ewc_lambda: float = 1000.0):
        """
        Args:
            checkpoint_path: Path to the trained offline model checkpoint
            device: Device to load the model on
            enable_ewc: Whether to enable EWC support
            ewc_lambda: EWC regularization strength
        """
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.model           = None
        self.config          = None
        self.replay_buffers  = {}
        self.available_tasks = []
        self.current_task_id = None
        
        # EWC components
        self.enable_ewc     = enable_ewc
        self.fisher_manager = None
        self.ewc_loss       = None
        
        # Load the model from checkpoint
        self._load_model_from_checkpoint()
        
        # Initialize EWC if enabled
        if self.enable_ewc:
            self._initialize_ewc(ewc_lambda)
        
    def _load_model_from_checkpoint(self):
        """
        Load the trained model and all associated data from the checkpoint.
        """
        print(f"[LOAD] Loading trained model from: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        try:
            # Import necessary classes
            from main_algorithm_v2.offline.src.config    import ExperimentConfig
            from main_algorithm_v2.offline.src.model     import UNet_SRCNN_LoRA
            from main_algorithm_v2.offline.replay_buffer import ReplayBuffer
            from main_algorithm_v2.offline.src.lora      import LoRAConv2d
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Report checkpoint contents
            # print(f"[INFO] Checkpoint contains keys: {list(checkpoint.keys())}")
            
            # Verify checkpoint structure
            required_keys = ['model_state_dict', 'config', 'replay_buffers', 'fisher_manager']
            missing_keys = []
            for key in required_keys:
                if key not in checkpoint:
                    missing_keys.append(key)

            #print(checkpoint['model_state_dict'].keys())
            
            if missing_keys:
                print(f"[WARNING] Missing keys in checkpoint: {missing_keys}")
                if 'config' in missing_keys:
                    print("[HINT] Ensure offline_config_path is specified in online config if config is not in checkpoint")
                raise KeyError(f"Required keys not found in checkpoint: {missing_keys}")
            
            # Extract config
            if isinstance(checkpoint['config'], dict):
                # If config is a dict, reconstruct ExperimentConfig
                self.config = ExperimentConfig.model_validate(checkpoint['config'])
                print(f"[SUCCESS] Config loaded from checkpoint (dict format)")
            else:
                self.config = checkpoint['config']
                print(f"[SUCCESS] Config loaded from checkpoint (object format)")
            
            print(f"[INFO] Model class: {checkpoint.get('model_class', 'Unknown')}")
            print(f"[INFO] Checkpoint type: {checkpoint.get('checkpoint_type', 'Unknown')}")
            
            # Initialize model with the loaded config
            self.model = UNet_SRCNN_LoRA(self.config).to(self.device)
            
            # Extract task IDs from checkpoint state dict keys
            state_dict = checkpoint['model_state_dict']
            task_ids = set()
            for key in state_dict.keys():
                if '.task_adapters.' in key:
                    # Extract task ID from keys like "backbone.unet.encoder_convs.0.0.task_adapters.0.A"
                    parts = key.split('.task_adapters.')
                    if len(parts) > 1:
                        task_id = parts[1].split('.')[0]  # Extract the task ID
                        task_ids.add(task_id)
            
            task_ids = sorted(list(task_ids))
            print(f"[INFO] Found LoRA adapters for tasks: {task_ids}")
            
            # Debug: Print actual adapter shapes to understand the real rank values
            print("\n[DEBUG] Actual adapter tensor shapes from checkpoint:")
            for task_id in task_ids:
                print(f"Task {task_id}:")
                first_layer_found = False
                for key, tensor in state_dict.items():
                    if f'.task_adapters.{task_id}.' in key and not first_layer_found:
                        if '.A' in key:
                            a_shape = tensor.shape
                            # Find corresponding B tensor
                            b_key = key.replace('.A', '.B')
                            if b_key in state_dict:
                                b_shape = state_dict[b_key].shape
                                rank = a_shape[0]  # First dimension of A is the rank
                                print(f"  Layer {key.split('task_adapters.')[0]}task_adapters.{task_id}: A{a_shape}, B{b_shape}, rank={rank}")
                                first_layer_found = True
                        break
            print("")
            
            # Extract the correct r and alpha values from adapter tensor shapes 
            # and create adapter structure with correct parameters
            print("[INFO] Creating adapter structures with correct r and alpha from checkpoint...")
            for task_id in task_ids:
                # Find first adapter to get the rank
                sample_a_tensor = None
                sample_layer_path = None
                for key, tensor in state_dict.items():
                    if f'.task_adapters.{task_id}.A' in key:
                        sample_a_tensor = tensor
                        sample_layer_path = key.split('.task_adapters.')[0]
                        break
                
                if sample_a_tensor is not None:
                    # The rank is the first dimension of the A tensor
                    r = sample_a_tensor.shape[0]
                    
                    # Try to get alpha from config if available, otherwise use a reasonable default
                    try:
                        alpha = self.config.model.params.task_lora_alphas[int(task_id)]
                    except (KeyError, AttributeError, TypeError):
                        alpha = r  # Use rank as alpha if not found in config
                    
                    print(f"Task {task_id}: Using r={r}, alpha={alpha} (inferred from checkpoint)")
                    
                    # Manually add adapters to each LoRA layer with correct parameters
                    for module in self.model.modules():
                        if isinstance(module, LoRAConv2d):
                            module.add_task_adapters(task_id, r, alpha)
                else:
                    print(f"[WARNING] No adapter tensors found for task {task_id}, using model defaults")
                    self.model.add_task(task_id)
            
            # Now load the state dict - this should populate the LoRA adapter weights
            print("[INFO] Loading LoRA adapter weights...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            
            # Report any missing or unexpected keys (this is normal for structure differences)
            if missing_keys:
                print(f"[INFO] Missing keys in checkpoint: {len(missing_keys)} keys")
                # Don't print all keys to avoid clutter, but show a few examples
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"    - {key}")
                else:
                    print(f"    - {missing_keys[0]} (and {len(missing_keys)-1} more)")
                        
            if unexpected_keys:
                print(f"[INFO] Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                # Don't print all keys to avoid clutter
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"    - {key}")
                else:
                    print(f"    - {unexpected_keys[0]} (and {len(unexpected_keys)-1} more)")
            
            print("[SUCCESS] Model state loaded successfully (with structure adaptation)")
            
            # Extract available tasks from loaded LoRA adapters
            available_tasks = set()
            for module in self.model.modules():
                if hasattr(module, 'task_adapters'):  # This is a LoRAConv2d
                    available_tasks.update(module.task_adapters.keys())
            
            self.available_tasks = sorted(list(available_tasks))
            print(f"Available tasks from LoRA adapters: {self.available_tasks}")
            
            # Load replay buffers
            replay_buffer_states = checkpoint['replay_buffers']
            for task_id_str, buffer_state in replay_buffer_states.items():
                # Extract buffer size from the loaded state
                samples = buffer_state.get('samples', [])
                buffer_size = len(samples) if samples else 1000  # Default to 1000 if empty
                
                replay_buffer = ReplayBuffer(buffer_size=buffer_size)
                replay_buffer.load_state_dict(buffer_state)
                self.replay_buffers[task_id_str] = replay_buffer
                print(f"[SUCCESS] Replay buffer for task {task_id_str}: {len(replay_buffer)} samples")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Extract additional info if available
            if 'best_task_val_losses' in checkpoint:
                print("Best validation losses per domain:")
                for task_id, val_loss in checkpoint['best_task_val_losses'].items():
                    print(f"  Task {task_id}: {val_loss:.6f}")
                    
            print("[SUCCESS] Model manager initialised successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
    
    def _initialize_ewc(self, ewc_lambda: float):
        """
        Initialize EWC components by loading Fisher matrices from checkpoint.
        
        Args:
            ewc_lambda: EWC regularization strength
        """
        print(f"\n[EWC] Initializing EWC support...")
        
        try:
            # Create Fisher manager and load from checkpoint
            self.fisher_manager = create_online_ewc_manager(
                lambda_ewc=ewc_lambda,
                offline_checkpoint_path=self.checkpoint_path
            )
            
            # Create EWC loss module
            self.ewc_loss = OnlineEWCLoss(self.fisher_manager)
            
            print(f"[SUCCESS] EWC initialized successfully")
            print(f"   [INFO] Fisher matrices for tasks: {list(self.fisher_manager.fisher_matrices.keys())}")
            print(f"   [INFO] Lambda EWC: {self.fisher_manager.lambda_ewc}")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize EWC: {e}")
            print(f"   [INFO] Disabling EWC support...")
            self.enable_ewc = False
            self.fisher_manager = None
            self.ewc_loss = None
    
    def set_active_domain(self, domain_id: int):
        """
        Switch the model to the specified domain.
        
        Args:
            domain_id: Domain ID (0-8) to switch to
        """
        task_id_str = str(domain_id)
        
        if task_id_str not in self.available_tasks:
            raise ValueError(f"Domain {domain_id} not available. Available domains: {self.available_tasks}")
        
        # Switch task in the model (LoRA adapters + BN statistics)
        self.model.set_active_task(task_id_str)
        self.current_task_id = task_id_str
        
        # print(f"[SWITCH] Switched to domain {domain_id}")  # Removed for minimal output
    
    def get_current_domain(self) -> Optional[int]:
        """
        Get the currently active domain.
        
        Returns:
            Current domain ID or None if no domain is active
        """
        if self.current_task_id is not None:
            return int(self.current_task_id)
        return None
    
    def inference(self, model_input: torch.Tensor) -> torch.Tensor:
        """
        Perform inference with the current domain setup.
        
        Args:
            model_input: Tensor of shape (2, 72, 70) or (batch_size, 2, 72, 70)
            
        Returns:
            Model output tensor
        """
        if self.current_task_id is None:
            raise RuntimeError("No domain is currently active. Call set_active_domain() first.")
        
        # Ensure input has batch dimension
        if model_input.dim() == 3:
            model_input = model_input.unsqueeze(0)  # Add batch dimension
            remove_batch_dim = True
        else:
            remove_batch_dim = False
        
        # Move to device
        model_input = model_input.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(model_input)
        
        # Remove batch dimension if it was added
        if remove_batch_dim:
            output = output.squeeze(0)
        
        # Ensure output is on the correct device
        output = output.to(self.device)
            
        return output
    
    def timed_inference(self, model_input: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Perform inference with timing measurement.
        
        Args:
            model_input: Input tensor
            
        Returns:
            Tuple of (output, inference_time_seconds)
        """
        start_time = time.time()
        output = self.inference(model_input)
        inference_time = time.time() - start_time
        
        return output, inference_time
    
    def get_replay_buffer(self, domain_id: int) -> Optional[ReplayBuffer]:
        """
        Get the replay buffer for a specific domain.
        
        Args:
            domain_id: Domain ID
            
        Returns:
            ReplayBuffer or None if not found
        """
        task_id_str = str(domain_id)
        return self.replay_buffers.get(task_id_str, None)
    
    def get_replay_buffers(self) -> Dict[str, ReplayBuffer]:
        """
        Get all replay buffers.
        
        Returns:
            Dictionary of all replay buffers keyed by task_id
        """
        return self.replay_buffers
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'checkpoint_path': self.checkpoint_path,
            'available_domains': [int(tid) for tid in self.available_tasks],
            'current_domain': self.get_current_domain(),
            'device': str(self.device),
            'model_name': self.config.model.name if self.config else 'Unknown',
            'total_domains': len(self.available_tasks),
        }
        
        # Add replay buffer info
        buffer_info = {}
        for task_id, buffer in self.replay_buffers.items():
            buffer_info[task_id] = len(buffer)
        info['replay_buffer_sizes'] = buffer_info
        
        return info
    
    def get_normalisation_stats(self):
        """
        Get the normalisation statistics from the model config.
        
        Returns:
            NormStats object
        """
        if self.config and hasattr(self.config.data, 'norm_stats'):
            return self.config.data.norm_stats
        else:
            raise RuntimeError("Normalisation statistics not found in model config")
    
    def switch_domain_with_timing(self, domain_id: int) -> float:
        """
        Switch domain and measure the time taken.
        
        Args:
            domain_id: Domain to switch to
            
        Returns:
            Time taken for domain switching in seconds
        """
        start_time = time.time()
        self.set_active_domain(domain_id)
        switch_time = time.time() - start_time
        
        return switch_time
    
    def get_ewc_loss(self, current_task_id: str, exclude_current_task: bool = True) -> torch.Tensor:
        """
        Compute EWC regularization loss for the current model state.
        
        Args:
            current_task_id: Current task being trained
            exclude_current_task: Whether to exclude current task from penalty
            
        Returns:
            EWC loss tensor (0 if EWC is disabled)
        """
        if not self.enable_ewc or self.ewc_loss is None:
            return torch.tensor(0.0, device=self.device)
        
        return self.ewc_loss(self.model, current_task_id, exclude_current_task)
    
    def get_fisher_manager(self) -> Optional[OnlineFisherInformationManager]:
        """
        Get the Fisher Information Manager.
        
        Returns:
            Fisher manager or None if EWC is disabled
        """
        return self.fisher_manager
    
    def is_ewc_enabled(self) -> bool:
        """
        Check if EWC is enabled and properly initialized.
        
        Returns:
            True if EWC is enabled and working
        """
        return self.enable_ewc and self.fisher_manager is not None and self.ewc_loss is not None
    
    def get_ewc_info(self) -> Dict[str, Any]:
        """
        Get information about EWC state.
        
        Returns:
            Dictionary with EWC information
        """
        if not self.enable_ewc:
            return {'enabled': False, 'reason': 'EWC disabled'}
        
        if self.fisher_manager is None or self.ewc_loss is None:
            return {'enabled': False, 'reason': 'EWC initialization failed'}
        
        return {
            'enabled': True,
            'lambda_ewc': self.fisher_manager.lambda_ewc,
            'tasks_with_fisher': list(self.fisher_manager.fisher_matrices.keys()),
            'total_fisher_parameters': sum(
                sum(f.numel() for f in task_fisher.values()) 
                for task_fisher in self.fisher_manager.fisher_matrices.values()
            )
        } 