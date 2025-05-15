
import torch.nn as nn
import torch.optim as optim


def get_criterion(config):
    loss_fn_name = config.get('training', {}).get('loss_function', 'mse').lower()
    if loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'mae' or loss_fn_name == 'l1':
        return nn.L1Loss()
    elif loss_fn_name == 'huber':
        delta = float(config.get('training', {}).get('huber_delta', 1.0))
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}. Supported: 'mse', 'mae', 'l1', 'huber'")
    

def get_optimiser(config, model):
    """Get the optimiser based on the config."""
    optim_config = config.get('training', {})
    optim_name   = optim_config.get('optimiser', 'adam').lower()
    lr           = float(optim_config.get('learning_rate', 1e-4))
    weight_decay = float(optim_config.get('weight_decay', 0))
    params_to_optimise = filter(lambda p: p.requires_grad, model.parameters())

    if optim_name == 'adam':
        betas = optim_config.get('betas', [0.9, 0.999])
        eps   = float(optim_config.get('eps', 1e-8))
        return optim.Adam(params_to_optimise, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        momentum = float(optim_config.get('momentum', 0.9))
        return optim.SGD(params_to_optimise, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimiser: {optim_name}")


def get_scheduler(config, optimiser):
    """Get the scheduler based on the config."""
    sched_config = config.get('training', {})
    if not sched_config.get('scheduler', False):
        return None
    
    scheduler_type = sched_config.get('scheduler_type', 'ReduceLROnPlateau').lower()
    
    if scheduler_type == 'reducelronplateau':
        factor   = float(sched_config.get('scheduler_factor', 0.1))
        patience = int(sched_config.get('scheduler_patience', 10))
        min_lr   = float(sched_config.get('scheduler_min_lr', 1e-6))
        return optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    else:
        print(f"Warning: Unsupported scheduler type '{scheduler_type}'. No scheduler will be used.")
        return None