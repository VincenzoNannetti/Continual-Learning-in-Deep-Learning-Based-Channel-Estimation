#!/usr/bin/env python3
"""Debug script to test evaluation and catch errors."""

import sys
import traceback
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_evaluation():
    try:
        # Try importing everything first
        print("Testing imports...")
        from src.config import ExperimentConfig
        from src.utils import load_lora_model_for_evaluation, get_device
        print("✓ Imports successful")
        
        # Test loading a checkpoint
        checkpoint_path = "checkpoints/unet_srcnn_lora_refactored-20250607_112601.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            # List available checkpoints
            if os.path.exists("checkpoints"):
                checkpoints = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
                print(f"Available checkpoints: {checkpoints}")
            return
        
        print(f"Testing checkpoint loading: {checkpoint_path}")
        device = get_device('cpu')  # Use CPU to avoid GPU issues
        model = load_lora_model_for_evaluation(checkpoint_path, device)
        print("✓ Model loading successful")
        
        # Test domain switching
        config = model.config
        test_task = str(config.data.sequence[0])
        print(f"Testing domain switching to task: {test_task}")
        model.set_active_task(test_task)
        print("✓ Domain switching successful")
        
        print("\n🎉 All tests passed! Evaluation should work.")
        
    except Exception as e:
        print(f"\n❌ Error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation() 