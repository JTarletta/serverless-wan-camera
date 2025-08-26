import runpod
import logging
import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import threading
import sys

from utils.workflow import process_image_to_video
from utils.model_manager import check_and_download_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global ComfyUI process
comfy_process = None

def start_comfyui_server():
    """Start ComfyUI server in background"""
    global comfy_process
    
    if comfy_process is not None and comfy_process.poll() is None:
        logger.info("ComfyUI server already running")
        return
    
    logger.info("Starting ComfyUI server...")
    
    # Change to ComfyUI directory
    comfy_dir = Path("/app/ComfyUI")
    if not comfy_dir.exists():
        raise Exception("ComfyUI directory not found at /app/ComfyUI")
    
    # Start ComfyUI with GPU support
    cmd = [
        "python3", "main.py",
        "--listen", "0.0.0.0",
        "--port", "8188", 
        "--force-fp16",  # Use FP16 to save VRAM
        "--disable-auto-launch",
        "--disable-metadata"
    ]
    
    try:
        # Set environment variables for CUDA
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        comfy_process = subprocess.Popen(
            cmd,
            cwd=comfy_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Wait a bit for server to start
        time.sleep(15)
        
        # Check if process is still running
        if comfy_process.poll() is not None:
            # Read any error output
            try:
                stdout, stderr = comfy_process.communicate(timeout=5)
                logger.error(f"ComfyUI failed to start. Output: {stdout}")
            except:
                pass
            raise Exception("ComfyUI failed to start")
        
        logger.info("ComfyUI server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        raise

def wait_for_comfyui():
    """Wait for ComfyUI to be ready"""
    import requests
    
    max_retries = 60  # Increased timeout
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8188", timeout=10)
            if response.status_code == 200:
                logger.info("✓ ComfyUI is ready")
                return True
        except Exception as e:
            if i == 0:
                logger.info("Waiting for ComfyUI to start...")
            elif i % 10 == 0:  # Log every 20 seconds
                logger.info(f"Still waiting for ComfyUI... ({i+1}/{max_retries})")
        
        time.sleep(2)
    
    raise Exception(f"ComfyUI failed to become ready after {max_retries * 2} seconds")

def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters"""
    
    # Required fields
    if "image" not in job_input:
        raise ValueError("Missing required field: image")
    
    # Extract and validate image
    image_data = job_input["image"]
    if isinstance(image_data, str):
        # Assume it's base64 encoded
        try:
            # Check if it has data URL prefix and remove it
            if image_data.startswith('data:image/'):
                image_data = image_data.split(',', 1)[1]
            base64.b64decode(image_data)
        except Exception:
            raise ValueError("Invalid base64 image data")
    else:
        raise ValueError("Image must be base64 encoded string")
    
    # Set defaults for optional parameters
    validated = {
        "image": image_data,
        "prompt": job_input.get("prompt", "cinematic static objects, smooth camera movement"),
        "camera_type": job_input.get("camera_type", "Zoom In"),
        "width": job_input.get("width", 832),
        "height": job_input.get("height", 448), 
        "length": job_input.get("length", 93),  # frames
        "speed": job_input.get("speed", 0.2),   # camera speed
        "fps": job_input.get("fps", 30),        # video fps
    }
    
    # Validate camera type
    valid_camera_types = ["Zoom In", "Static", "Zoom Out", "Pan Left", "Pan Right"]
    if validated["camera_type"] not in valid_camera_types:
        raise ValueError(f"Invalid camera_type. Must be one of: {valid_camera_types}")
    
    # Validate dimensions
    if not (256 <= validated["width"] <= 1920 and 256 <= validated["height"] <= 1920):
        raise ValueError("Width and height must be between 256 and 1920")
    
    # Validate length
    if not (16 <= validated["length"] <= 200):
        raise ValueError("Length must be between 16 and 200 frames")
        
    # Validate speed
    if not (0.1 <= validated["speed"] <= 1.0):
        raise ValueError("Speed must be between 0.1 and 1.0")
    
    return validated

def handler(job):
    """Main RunPod handler function"""
    job_input = job["input"]
    job_id = job.get('id', 'unknown')
    
    try:
        logger.info(f"🎬 Processing job: {job_id}")
        logger.info(f"Input parameters: {list(job_input.keys())}")
        
        # Validate input
        validated_input = validate_input(job_input)
        logger.info("✓ Input validation passed")
        
        # Process image to video
        start_time = time.time()
        video_data, filename = process_image_to_video(**validated_input)
        processing_time = time.time() - start_time
        
        # Encode video as base64 for response
        video_b64 = base64.b64encode(video_data).decode('utf-8')
        
        logger.info(f"✓ Video generation completed in {processing_time:.1f}s")
        logger.info(f"✓ Output filename: {filename}")
        logger.info(f"✓ Video size: {len(video_data) / (1024*1024):.1f} MB")
        
        return {
            "video": video_b64,
            "filename": filename,
            "processing_time": processing_time,
            "parameters_used": validated_input,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}

def initialize():
    """Initialize the serverless function"""
    logger.info("🚀 Initializing Wan Camera Serverless...")
    
    # Check volume mount
    volume_path = os.environ.get('RUNPOD_VOLUME_PATH', '/runpod-volume')
    if not Path(volume_path).exists():
        raise Exception(f"Volume not mounted at {volume_path}")
    
    logger.info(f"✓ Volume mounted at: {volume_path}")
    
    # Download models if needed
    logger.info("📦 Checking models...")
    check_and_download_models()
    logger.info("✓ Models ready")
    
    # Start ComfyUI server
    start_comfyui_server()
    
    # Wait for ComfyUI to be ready
    wait_for_comfyui()
    
    logger.info("🎉 Initialization complete - Ready for requests!")

if __name__ == "__main__":
    try:
        # Initialize
        initialize()
        
        # Start RunPod serverless
        logger.info("🔄 Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}", exc_info=True)
        sys.exit(1)