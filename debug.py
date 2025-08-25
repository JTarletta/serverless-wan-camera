#!/usr/bin/env python3
"""
Debug script to test components individually
Run this in the pod to verify everything works
"""

import sys
import subprocess
import time
from pathlib import Path

def check_volume():
    """Check if volume is mounted correctly"""
    volume_path = Path("/serverless_wan2_vol")
    print(f"Checking volume at {volume_path}...")
    
    if volume_path.exists():
        print(f"✓ Volume mounted at {volume_path}")
        return True
    else:
        print(f"✗ Volume NOT found at {volume_path}")
        return False

def check_models():
    """Check model download"""
    try:
        from utils.model_manager import check_and_download_models
        print("Checking and downloading models...")
        check_and_download_models()
        print("✓ Models check completed")
        return True
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        return False

def start_comfyui():
    """Start ComfyUI manually for testing"""
    print("Starting ComfyUI...")
    try:
        cmd = [
            "python3", "/app/ComfyUI/main.py",
            "--listen", "0.0.0.0",
            "--port", "8188",
            "--force-fp16"
        ]
        
        process = subprocess.Popen(cmd)
        print(f"ComfyUI started with PID: {process.pid}")
        print("Wait ~30 seconds, then visit: http://localhost:8188")
        return True
    except Exception as e:
        print(f"✗ ComfyUI start failed: {e}")
        return False

def test_workflow():
    """Test basic workflow creation"""
    try:
        from utils.workflow import create_wan_workflow
        
        workflow = create_wan_workflow(
            image_filename="test.png",
            prompt="test prompt",
            camera_type="Zoom In"
        )
        
        print(f"✓ Workflow created with {len(workflow)} nodes")
        return True
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== WAN CAMERA DEBUG SCRIPT ===")
    
    tests = [
        ("Volume Mount", check_volume),
        ("Model Download", check_models),
        ("Workflow Creation", test_workflow),
        ("ComfyUI Start", start_comfyui)
    ]
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        if not success:
            print(f"Failed at: {test_name}")
            sys.exit(1)
    
    print("\n✓ All tests passed!")
    print("ComfyUI should be running on port 8188")