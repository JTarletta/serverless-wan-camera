import os
import requests
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs and file info
MODELS_CONFIG = {
    "diffusion_models": {
        "wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors",
            "size": 17 * 1024 * 1024 * 1024  # 17 GB
        },
        "wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors", 
            "size": 17 * 1024 * 1024 * 1024  # 17 GB
        }
    },
    "loras": {
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "size": 614 * 1024 * 1024  # 614 MB
        },
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "size": 614 * 1024 * 1024  # 614 MB
        }
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "size": int(6.74 * 1024 * 1024 * 1024)  # 6.74 GB
        }
    },
    "vae": {
        "wan_2.1_vae.safetensors": {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
            "size": 254 * 1024 * 1024  # 254 MB
        }
    }
}

def download_file(url: str, filepath: Path, expected_size: int = None) -> bool:
    """Download a file with progress tracking"""
    try:
        logger.info(f"Downloading {filepath.name}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total size from headers
        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            logger.warning(f"Size mismatch: expected {expected_size}, got {total_size}")
        
        downloaded = 0
        chunk_size = 8192 * 16  # 128KB chunks
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every ~100MB
                    if downloaded % (100 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        logger.info(f"  Progress: {downloaded / (1024*1024*1024):.1f}GB ({progress:.1f}%)")
        
        logger.info(f"✓ Downloaded {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {filepath.name}: {e}")
        # Remove partial file
        if filepath.exists():
            filepath.unlink()
        return False

def check_and_download_models() -> None:
    """Check for missing models and download them"""
    
    volume_path = Path(os.environ.get('RUNPOD_VOLUME_PATH', '/runpod-volume'))
    models_base = volume_path / 'models'
    
    logger.info("Checking model files...")
    
    missing_models = []
    total_missing_size = 0
    
    # Check each model category
    for category, models in MODELS_CONFIG.items():
        category_path = models_base / category
        
        for filename, info in models.items():
            filepath = category_path / filename
            
            if not filepath.exists():
                missing_models.append((category, filename, info))
                total_missing_size += info['size']
                logger.info(f"Missing: {category}/{filename}")
            else:
                logger.info(f"Found: {category}/{filename}")
    
    if not missing_models:
        logger.info("All models present!")
        return
    
    # Log total download size
    total_gb = total_missing_size / (1024 * 1024 * 1024)
    logger.info(f"Need to download {len(missing_models)} files ({total_gb:.1f} GB total)")
    
    # Download missing models
    for category, filename, info in missing_models:
        category_path = models_base / category
        filepath = category_path / filename
        
        success = download_file(info['url'], filepath, info['size'])
        if not success:
            raise Exception(f"Failed to download required model: {filename}")
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    check_and_download_models()