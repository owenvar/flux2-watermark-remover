"""
RunPod Serverless Handler for FLUX.2 Dev Image Editing
Optimized for watermark removal using 4-bit quantized model
"""

import runpod
import torch
import base64
import io
import os
import requests
from PIL import Image

# Global pipeline (loaded once, reused across requests)
pipe = None
remote_encoder = None


def get_hf_token():
    """Get HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set!")
    return token


def encode_prompt_remote(prompt):
    """Encode prompt using HuggingFace's remote text encoder."""
    if isinstance(prompt, str):
        prompt = [prompt]
    
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {get_hf_token()}",
            "Content-Type": "application/json"
        },
        timeout=60
    )
    response.raise_for_status()
    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds


def load_model():
    """Load the FLUX.2 Dev model with 4-bit quantization."""
    global pipe
    
    if pipe is not None:
        return pipe
    
    print("Loading FLUX.2 Dev model (4-bit quantized)...")
    
    from diffusers import Flux2Pipeline
    
    # Use 4-bit quantized model for memory efficiency
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    
    pipe = Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=None,  # Use remote text encoder to save VRAM
        torch_dtype=torch.bfloat16,
        token=get_hf_token()
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    print("Model loaded successfully!")
    return pipe


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    # Remove data URI prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGB")


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(event):
    """
    RunPod handler for FLUX.2 Dev image editing.
    
    Input format:
    {
        "input": {
            "image": "base64_encoded_image",
            "prompt": "Remove the watermark...",
            "steps": 28,
            "guidance": 4.0,
            "seed": 42
        }
    }
    """
    try:
        # Load model (cached after first call)
        pipeline = load_model()
        
        # Get input parameters
        job_input = event.get("input", {})
        
        # Required: image
        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "Missing required 'image' parameter"}
        
        # Decode input image
        input_image = decode_base64_image(image_base64)
        original_size = input_image.size
        print(f"Input image size: {original_size}")
        
        # Get prompt (default: watermark removal)
        prompt = job_input.get("prompt", 
            "Remove all watermark text, logo, and phone number from the image. "
            "Replace the watermark area with the natural background. "
            "Keep everything else exactly the same. Do not change the main subject."
        )
        
        # Optional parameters
        steps = job_input.get("steps", 28)
        guidance = job_input.get("guidance", 4.0)
        seed = job_input.get("seed", None)
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        
        print(f"Prompt: {prompt[:80]}...")
        print(f"Steps: {steps}, Guidance: {guidance}")
        
        # Encode prompt using remote text encoder
        print("Encoding prompt...")
        prompt_embeds = encode_prompt_remote(prompt).to("cuda")
        
        # Run inference with input image as reference
        print("Running inference...")
        output = pipeline(
            prompt_embeds=prompt_embeds,
            image=[input_image],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            height=original_size[1],
            width=original_size[0],
        )
        
        # Get output image
        result_image = output.images[0]
        
        # Encode to base64
        result_base64 = encode_image_to_base64(result_image)
        
        print(f"Done! Output size: {result_image.size}")
        
        return {
            "images": [
                {
                    "data": result_base64,
                    "format": "png",
                    "width": result_image.width,
                    "height": result_image.height
                }
            ]
        }
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
