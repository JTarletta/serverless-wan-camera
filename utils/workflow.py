import json
import uuid
import requests
import websocket
import threading
import time
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class ComfyUIWorkflow:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a prompt and return the prompt ID"""
        try:
            prompt = {"prompt": workflow, "client_id": self.client_id}
            data = json.dumps(prompt).encode('utf-8')
            
            response = requests.post(f"http://{self.server_address}/prompt", data=data)
            response.raise_for_status()
            
            result = response.json()
            return result['prompt_id']
            
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            raise

    def upload_image(self, image_data: bytes, filename: str) -> bool:
        """Upload image to ComfyUI input directory"""
        try:
            files = {'image': (filename, image_data, 'image/png')}
            response = requests.post(f"http://{self.server_address}/upload/image", files=files)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            raise

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Get generated image from ComfyUI"""
        try:
            data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            url = f"http://{self.server_address}/view"
            
            response = requests.get(url, params=data)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to get image: {e}")
            raise

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        try:
            response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            raise

    def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for prompt completion and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
                
                if prompt_id in history:
                    result = history[prompt_id]
                    
                    # Check if execution is complete
                    if 'outputs' in result:
                        return result
                        
                    # Check for errors
                    if result.get('status', {}).get('status_str') == 'error':
                        raise Exception(f"ComfyUI execution failed: {result['status']}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except requests.exceptions.RequestException:
                # ComfyUI might not be fully ready, wait a bit
                time.sleep(5)
                
        raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")

# Load the workflow template (your exported JSON)
WORKFLOW_TEMPLATE = {
    "71": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 316038771312063,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["76", 0],
            "positive": ["80", 0],
            "negative": ["80", 1],
            "latent_image": ["80", 2]
        },
        "class_type": "KSamplerAdvanced"
    },
    "72": {
        "inputs": {
            "unet_name": "wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader"
    },
    "73": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video": ["83", 0]
        },
        "class_type": "SaveVideo"
    },
    "74": {
        "inputs": {
            "text": "static image, no movement, oversaturated colors, overexposed, blurry details, poor quality, jpeg artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, mutated limbs, fused fingers, frozen frame, cluttered background, three legs, crowded background, walking backwards, abrupt cuts, jerky movement, low resolution, pixelated, grainy, washed out colors, flat lighting, amateur quality, shaky camera",
            "clip": ["85", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "75": {
        "inputs": {
            "unet_name": "wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader"
    },
    "76": {
        "inputs": {
            "shift": 8.000000000000002,
            "model": ["88", 0]
        },
        "class_type": "ModelSamplingSD3"
    },
    "77": {
        "inputs": {
            "shift": 8,
            "model": ["90", 0]
        },
        "class_type": "ModelSamplingSD3"
    },
    "78": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["77", 0],
            "positive": ["80", 0],
            "negative": ["80", 1],
            "latent_image": ["71", 0]
        },
        "class_type": "KSamplerAdvanced"
    },
    "79": {
        "inputs": {
            "image": "amueblar_after.png"
        },
        "class_type": "LoadImage"
    },
    "80": {
        "inputs": {
            "width": ["87", 1],
            "height": ["87", 2],
            "length": ["87", 3],
            "batch_size": 1,
            "positive": ["81", 0],
            "negative": ["74", 0],
            "vae": ["86", 0],
            "start_image": ["79", 0],
            "camera_conditions": ["87", 0]
        },
        "class_type": "WanCameraImageToVideo"
    },
    "81": {
        "inputs": {
            "text": "cinematic static objects, smooth camera movement, atmospheric lighting",
            "clip": ["85", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "82": {
        "inputs": {
            "samples": ["78", 0],
            "vae": ["86", 0]
        },
        "class_type": "VAEDecode"
    },
    "83": {
        "inputs": {
            "fps": 30,
            "images": ["82", 0]
        },
        "class_type": "CreateVideo"
    },
    "85": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default"
        },
        "class_type": "CLIPLoader"
    },
    "86": {
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors"
        },
        "class_type": "VAELoader"
    },
    "87": {
        "inputs": {
            "camera_pose": "Zoom In",
            "width": 832,
            "height": 448,
            "length": 93,
            "speed": 0.2,
            "fx": 0.5,
            "fy": 0.5,
            "cx": 0.5,
            "cy": 0.5
        },
        "class_type": "WanCameraEmbedding"
    },
    "88": {
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "strength_model": 1,
            "model": ["75", 0]
        },
        "class_type": "LoraLoaderModelOnly"
    },
    "90": {
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "strength_model": 1,
            "model": ["72", 0]
        },
        "class_type": "LoraLoaderModelOnly"
    }
}

def create_wan_workflow(
    image_filename: str,  # Image filename (será subida a ComfyUI)
    prompt: str,
    camera_type: str = "Zoom In",
    width: int = 832,
    height: int = 448,
    length: int = 93,
    **kwargs
) -> Dict[str, Any]:
    """Create workflow by modifying the template with user parameters"""
    import copy
    
    # Deep copy the template to avoid modifying the original
    workflow = copy.deepcopy(WORKFLOW_TEMPLATE)
    
    # Update dynamic parameters
    workflow["79"]["inputs"]["image"] = image_filename  # LoadImage
    workflow["81"]["inputs"]["text"] = prompt  # Positive prompt
    workflow["87"]["inputs"]["camera_pose"] = camera_type  # Camera type
    workflow["87"]["inputs"]["width"] = width
    workflow["87"]["inputs"]["height"] = height  
    workflow["87"]["inputs"]["length"] = length
    
    return workflow

def process_image_to_video(
    image_data: str,  # base64 encoded image
    prompt: str, 
    camera_type: str = "Zoom In",
    width: int = 832,
    height: int = 448,
    length: int = 93,
    **kwargs
) -> Tuple[bytes, str]:
    """Process image to video using ComfyUI workflow"""
    
    comfy = ComfyUIWorkflow()
    
    # Convert base64 to bytes and upload to ComfyUI
    import base64
    import uuid
    
    image_bytes = base64.b64decode(image_data)
    temp_filename = f"input_{uuid.uuid4().hex[:8]}.png"
    
    # Upload image to ComfyUI
    comfy.upload_image(image_bytes, temp_filename)
    logger.info(f"Uploaded image as: {temp_filename}")
    
    # Create workflow using the template
    workflow = create_wan_workflow(
        image_filename=temp_filename,
        prompt=prompt,
        camera_type=camera_type,
        width=width,
        height=height,
        length=length,
        **kwargs
    )
    
    # Queue and execute
    prompt_id = comfy.queue_prompt(workflow)
    logger.info(f"Queued workflow with prompt_id: {prompt_id}")
    
    # Wait for completion
    result = comfy.wait_for_completion(prompt_id)
    
    # Extract video file from results
    outputs = result.get('outputs', {})
    
    # Find the SaveVideo node output (node 73 in our workflow)
    video_output = None
    if '73' in outputs and 'videos' in outputs['73']:
        video_output = outputs['73']['videos'][0]
    
    if not video_output:
        raise Exception("No video output found in workflow result")
    
    # Get video file
    video_data = comfy.get_image(
        filename=video_output['filename'],
        subfolder=video_output.get('subfolder', ''),
        folder_type=video_output.get('type', 'output')
    )
    
    return video_data, video_output['filename']