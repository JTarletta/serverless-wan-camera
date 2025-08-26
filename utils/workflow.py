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

# Load the API workflow template from JSON file
def load_workflow_template():
    """Load the workflow template from the API JSON export"""
    import os
    workflow_path = os.path.join(os.path.dirname(__file__), '..', 'serverlessAPI_wan2_2_14B_camera.json')
    try:
        with open(workflow_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Workflow template not found at {workflow_path}")
        raise

WORKFLOW_TEMPLATE = load_workflow_template()

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
    length: int = 93,  # frames
    speed: float = 0.2,  # camera speed (0.1-1.0) 
    fps: int = 30,  # video fps
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
        speed=speed,
        fps=fps,
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