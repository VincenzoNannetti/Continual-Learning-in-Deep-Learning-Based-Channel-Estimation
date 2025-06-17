"""
Continual Learning Methods Module
"""

from .ewc import EWCTrainer
from .base_trainer import BaseContinualTrainer

__all__ = ['EWCTrainer', 'BaseContinualTrainer']

# TODO: Add these imports when methods are implemented:
# from .replay import ReplayTrainer  
# from .lwf import LwFTrainer 