import os
import requests
import json
import time
from pathlib import Path

# RunPod API Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise ValueError("Please set RUNPOD_API_KEY environment variable")

# Configuration
TEMPLATE_NAME = "llm-ad-integration"
GPU_TYPE = "NVIDIA A100 80GB"  # or whatever GPU you want to use
DOCKER_IMAGE = "your-dockerhub-username/llm-ad-integration:latest"  # Replace with your Docker Hub username

def create_template():
    """Create a RunPod template for the deployment."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    template_data = {
        "name": TEMPLATE_NAME,
        "imageName": DOCKER_IMAGE,
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "ports": "",
        "isPublic": False,
        "dockerArgs": "",
        "volumeMountPath": "/workspace",
        "env": [
            {"key": "DEEPSEEK_API_KEY", "value": os.getenv("DEEPSEEK_API_KEY", "")},
            {"key": "OPENAI_API_KEY", "value": os.getenv("OPENAI_API_KEY", "")}
        ]
    }
    
    response = requests.post(
        "https://api.runpod.ai/v2/templates",
        headers=headers,
        json=template_data
    )
    response.raise_for_status()
    return response.json()["id"]

def create_pod(template_id):
    """Create a RunPod instance using the template."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    pod_data = {
        "name": f"{TEMPLATE_NAME}-pod",
        "imageName": DOCKER_IMAGE,
        "gpuTypeId": GPU_TYPE,
        "cloudType": "SECURE",
        "templateId": template_id,
        "containerDiskInGb": 20
    }
    
    response = requests.post(
        "https://api.runpod.ai/v2/pods",
        headers=headers,
        json=pod_data
    )
    response.raise_for_status()
    return response.json()["id"]

def main():
    print("Creating RunPod template...")
    template_id = create_template()
    print(f"Template created with ID: {template_id}")
    
    print("\nCreating RunPod instance...")
    pod_id = create_pod(template_id)
    print(f"Pod created with ID: {pod_id}")
    print("\nYou can monitor your pod at: https://www.runpod.io/console/pods")

if __name__ == "__main__":
    main() 