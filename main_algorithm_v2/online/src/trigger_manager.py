"""
Trigger Manager for Online Continual Learning

This module implements different trigger strategies to determine when to perform
network updates during online continual learning. Supports time-based, volume-based,
drift detection, and hybrid triggers.
"""

import time
from typing import Dict, Any
from enum import Enum
from .loss_functions import exponential_moving_average


class TriggerType(Enum):
    """Types of triggers available."""
    TIME = "time"
    VOLUME = "volume"
    HYBRID = "hybrid"
    DRIFT = "drift"


class BaseTrigger:
    """Base class for all trigger implementations."""
    
    def __init__(self, name: str):
        self.name = name
        self.trigger_count = 0
        self.creation_time = time.time()
    
    def should_trigger(self, **kwargs) -> bool:
        """
        Determine if an update should be triggered.
        
        Returns:
            bool: True if update should be triggered
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset trigger state."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics."""
        return {
            'name': self.name,
            'trigger_count': self.trigger_count,
            'age_seconds': time.time() - self.creation_time
        }


class TimeTrigger(BaseTrigger):
    """Trigger based on time intervals."""
    
    def __init__(self, update_interval_seconds: float = 30.0):
        super().__init__("time_trigger")
        self.update_interval = update_interval_seconds
        self.last_update_time = time.time()
        
    def should_trigger(self, **kwargs) -> bool:
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            self.trigger_count += 1
            return True
        return False
    
    def reset(self):
        self.last_update_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            'update_interval_seconds': self.update_interval,
            'seconds_since_last_update': time.time() - self.last_update_time,
            'avg_trigger_interval': (time.time() - self.creation_time) / max(1, self.trigger_count)
        })
        return stats


class VolumeTrigger(BaseTrigger):
    """Trigger based on number of samples processed."""
    
    def __init__(self, samples_per_update: int = 50):
        super().__init__("volume_trigger")
        self.samples_per_update = samples_per_update
        self.samples_since_update = 0
        self.total_samples_processed = 0
        
    def should_trigger(self, **kwargs) -> bool:
        self.samples_since_update += 1
        self.total_samples_processed += 1
        
        if self.samples_since_update >= self.samples_per_update:
            self.samples_since_update = 0
            self.trigger_count += 1
            return True
        return False
    
    def reset(self):
        self.samples_since_update = 0
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            'samples_per_update': self.samples_per_update,
            'samples_since_update': self.samples_since_update,
            'total_samples_processed': self.total_samples_processed,
            'avg_samples_per_trigger': self.total_samples_processed / max(1, self.trigger_count)
        })
        return stats


class HybridTrigger(BaseTrigger):
    """Trigger based on both time and volume with configurable weights."""
    
    def __init__(self, max_time_seconds: float = 60.0, max_samples: int = 100, 
                 time_weight: float = 1.0, volume_weight: float = 1.0):
        super().__init__("hybrid_trigger")
        self.max_time = max_time_seconds
        self.max_samples = max_samples
        self.time_weight = time_weight
        self.volume_weight = volume_weight
        
        self.last_update_time = time.time()
        self.samples_since_update = 0
        self.total_samples_processed = 0
        
        # Statistics
        self.time_triggers = 0
        self.volume_triggers = 0
        self.hybrid_triggers = 0
        
    def should_trigger(self, **kwargs) -> bool:
        self.samples_since_update += 1
        self.total_samples_processed += 1
        
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        # Check individual conditions
        time_exceeded = time_elapsed >= self.max_time
        volume_exceeded = self.samples_since_update >= self.max_samples
        
        # Hybrid scoring (normalised between 0 and 1)
        time_score = min(1.0, time_elapsed / self.max_time) * self.time_weight
        volume_score = min(1.0, self.samples_since_update / self.max_samples) * self.volume_weight
        
        should_trigger = False
        trigger_reason = None
        
        if time_exceeded and volume_exceeded:
            should_trigger = True
            trigger_reason = "hybrid"
            self.hybrid_triggers += 1
        elif time_exceeded:
            should_trigger = True
            trigger_reason = "time"
            self.time_triggers += 1
        elif volume_exceeded:
            should_trigger = True
            trigger_reason = "volume"
            self.volume_triggers += 1
        
        if should_trigger:
            self.last_update_time = current_time
            self.samples_since_update = 0
            self.trigger_count += 1
            
        return should_trigger
    
    def reset(self):
        self.last_update_time = time.time()
        self.samples_since_update = 0
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        current_time = time.time()
        stats.update({
            'max_time_seconds': self.max_time,
            'max_samples': self.max_samples,
            'time_weight': self.time_weight,
            'volume_weight': self.volume_weight,
            'seconds_since_last_update': current_time - self.last_update_time,
            'samples_since_update': self.samples_since_update,
            'total_samples_processed': self.total_samples_processed,
            'time_triggers': self.time_triggers,
            'volume_triggers': self.volume_triggers,
            'hybrid_triggers': self.hybrid_triggers,
            'time_progress': min(1.0, (current_time - self.last_update_time) / self.max_time),
            'volume_progress': min(1.0, self.samples_since_update / self.max_samples)
        })
        return stats


class DriftTrigger(BaseTrigger):
    """Trigger based on performance drift detection using exponential moving average."""
    
    def __init__(self, alpha: float = 0.1, kappa: float = 1.5, warmup_samples: int = 50):
        """
        Initialize drift trigger.
        
        Args:
            alpha: Smoothing factor for exponential moving average (0 < α < 1)
            kappa: Drift sensitivity multiplier (threshold = κ × moving_average)
            warmup_samples: Number of samples before drift detection activates
        """
        super().__init__("drift_trigger")
        self.alpha = alpha
        self.kappa = kappa
        self.warmup_samples = warmup_samples
        
        # Moving average state
        self.moving_average = 0.0
        self.samples_processed = 0
        self.is_warmup_complete = False
        
        # Statistics
        self.drift_detections = 0
        self.loss_history = []
        self.threshold_history = []
        self.warmup_losses = []
        
    def should_trigger(self, current_loss: float, **kwargs) -> bool:
        """
        Determine if drift has been detected based on current loss.
        
        Args:
            current_loss: Current masked NMSE loss (L₀)
            
        Returns:
            bool: True if drift detected and update should be triggered
        """
        self.samples_processed += 1
        self.loss_history.append(current_loss)
        
        # Warmup phase - collect samples but don't trigger
        if not self.is_warmup_complete:
            self.warmup_losses.append(current_loss)
            
            if self.samples_processed >= self.warmup_samples:
                # Initialize moving average with warmup period average
                self.moving_average = sum(self.warmup_losses) / len(self.warmup_losses)
                self.is_warmup_complete = True
                print(f" Drift trigger warmup complete. Initial moving average: {self.moving_average:.6f}")
            
            return False
        
        # Calculate threshold
        threshold = self.kappa * self.moving_average
        self.threshold_history.append(threshold)
        
        # Check for drift
        drift_detected = current_loss > threshold
        
        if drift_detected:
            self.drift_detections += 1
            self.trigger_count += 1
            print(f" Drift detected! Loss: {current_loss:.6f} > Threshold: {threshold:.6f}")
        
        # Update moving average using exponential moving average
        self.moving_average = exponential_moving_average(
            current_ema=self.moving_average,
            new_value=current_loss,
            alpha=self.alpha
        )
        
        return drift_detected
    
    def reset(self):
        """Reset drift detection state (but keep learned moving average)."""
        # Don't reset moving_average or warmup state as they represent learned knowledge
        # Only reset immediate trigger state
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get drift trigger statistics."""
        stats = super().get_stats()
        
        recent_losses = self.loss_history[-10:] if self.loss_history else []
        recent_thresholds = self.threshold_history[-10:] if self.threshold_history else []
        
        stats.update({
            'alpha': self.alpha,
            'kappa': self.kappa,
            'warmup_samples': self.warmup_samples,
            'moving_average': self.moving_average,
            'samples_processed': self.samples_processed,
            'is_warmup_complete': self.is_warmup_complete,
            'drift_detections': self.drift_detections,
            'current_threshold': self.kappa * self.moving_average if self.is_warmup_complete else 0.0,
            'recent_losses': recent_losses,
            'recent_thresholds': recent_thresholds,
            'avg_loss': sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0,
            'latest_loss': self.loss_history[-1] if self.loss_history else 0.0,
        })
        
        return stats


class TriggerManager:
    """
    Manager for different trigger strategies in online continual learning.
    """
    
    def __init__(self, trigger_config: Dict[str, Any]):
        """
        Initialize trigger manager.
        
        Args:
            trigger_config: Configuration dict with keys:
                - type: str ("time", "volume", "hybrid", "drift")
                - time_interval: float (seconds for time-based)
                - volume_interval: int (samples for volume-based)
                - max_time: float (max seconds for hybrid)
                - max_samples: int (max samples for hybrid)
                - time_weight: float (weight for time component in hybrid)
                - volume_weight: float (weight for volume component in hybrid)
                - alpha: float (smoothing factor for drift detection)
                - kappa: float (drift sensitivity multiplier for drift detection)
                - warmup_samples: int (number of samples for warmup in drift detection)
        """
        self.config = trigger_config
        trigger_type = trigger_config.get('type', 'hybrid').lower()
        
        if trigger_type == TriggerType.TIME.value:
            self.trigger = TimeTrigger(
                update_interval_seconds=trigger_config.get('time_interval', 30.0)
            )
        elif trigger_type == TriggerType.VOLUME.value:
            self.trigger = VolumeTrigger(
                samples_per_update=trigger_config.get('volume_interval', 50)
            )
        elif trigger_type == TriggerType.HYBRID.value:
            self.trigger = HybridTrigger(
                max_time_seconds=trigger_config.get('max_time', 60.0),
                max_samples=trigger_config.get('max_samples', 100),
                time_weight=trigger_config.get('time_weight', 1.0),
                volume_weight=trigger_config.get('volume_weight', 1.0)
            )
        elif trigger_type == TriggerType.DRIFT.value:
            self.trigger = DriftTrigger(
                alpha=trigger_config.get('alpha', 0.1),
                kappa=trigger_config.get('kappa', 1.5),
                warmup_samples=trigger_config.get('warmup_samples', 50)
            )
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
        
        print(f" TriggerManager initialized with {trigger_type} trigger")
        print(f"   Config: {self.config}")
    
    def should_trigger_update(self, **kwargs) -> bool:
        """
        Check if an update should be triggered.
        
        Args:
            **kwargs: Additional arguments for specific trigger types.
                     For drift triggers, requires 'current_loss' parameter.
        
        Returns:
            bool: True if update should be triggered
        """
        return self.trigger.should_trigger(**kwargs)
    
    def reset_trigger(self):
        """Reset trigger state."""
        self.trigger.reset()
        #print(f" Trigger reset: {self.trigger.name}")
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get comprehensive trigger statistics."""
        return self.trigger.get_stats()
    
    def print_status(self):
        """Print current trigger status."""
        stats = self.get_trigger_stats()
        print(f"\n TRIGGER STATUS ({stats['name']}):")
        print("-" * 40)
        
        if stats['name'] == 'time_trigger':
            print(f"   Interval: {stats['update_interval_seconds']:.1f}s")
            print(f"   Since last: {stats['seconds_since_last_update']:.1f}s")
            print(f"   Avg interval: {stats['avg_trigger_interval']:.1f}s")
            
        elif stats['name'] == 'volume_trigger':
            print(f"   Samples per update: {stats['samples_per_update']}")
            print(f"   Current count: {stats['samples_since_update']}")
            print(f"   Total processed: {stats['total_samples_processed']}")
            
        elif stats['name'] == 'hybrid_trigger':
            print(f"   Time: {stats['seconds_since_last_update']:.1f}s / {stats['max_time_seconds']:.1f}s "
                  f"({stats['time_progress']:.1%})")
            print(f"   Volume: {stats['samples_since_update']} / {stats['max_samples']} "
                  f"({stats['volume_progress']:.1%})")
            print(f"   Triggers: T={stats['time_triggers']}, V={stats['volume_triggers']}, "
                  f"H={stats['hybrid_triggers']}")
        
        elif stats['name'] == 'drift_trigger':
            warmup_status = "Complete" if stats['is_warmup_complete'] else f"In Progress ({stats['samples_processed']}/{stats['warmup_samples']})"
            print(f"   Warmup: {warmup_status}")
            print(f"   Moving Average (ℓ̄_d): {stats['moving_average']:.6f}")
            print(f"   Current Threshold (κℓ̄_d): {stats['current_threshold']:.6f}")
            print(f"   Latest Loss: {stats['latest_loss']:.6f}")
            print(f"   Parameters: α={stats['alpha']:.2f}, κ={stats['kappa']:.2f}")
            print(f"   Drift Detections: {stats['drift_detections']}")
            print(f"   Samples Processed: {stats['samples_processed']}")
        
        print(f"   Total triggers: {stats['trigger_count']}")
        print(f"   Age: {stats['age_seconds']:.1f}s") 