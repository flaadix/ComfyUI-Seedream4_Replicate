import torch
import numpy as np
from PIL import Image
import io
import requests
import random
import os
import folder_paths
import replicate
import time
from colorama import Fore, Style, init
import logging
init(autoreset=True)

# Disable verbose HTTP logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("replicate").setLevel(logging.WARNING)
logging.getLogger("http.client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class Seedream4_Replicate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "placeholder": "Enter your Replicate API key",
                    "tooltip": "Your Replicate API token (starts with r8_). Get this from replicate.com/account/api-tokens. Keep this secure and don't share it."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Enter your prompt here...",
                    "tooltip": "Describe what you want to generate. Be specific and detailed. Example: 'a photorealistic portrait of a woman with blue eyes, studio lighting, high detail'"
                }),
                "size_preset": ([
                    "2048x2048 (1:1)",
                    "2304x1728 (4:3)", 
                    "1728x2304 (3:4)",
                    "2560x1440 (16:9)",
                    "1440x2560 (9:16)",
                    "2496x1664 (3:2)",
                    "1664x2496 (2:3)",
                    "3024x1296 (21:9)",
                    "4096x4096 (1:1)",
                    "Custom"
                ], {
                    "default": "2048x2048 (1:1)",
                    "tooltip": "Choose from common aspect ratios and sizes. Square formats work well for portraits and social media. Wide formats (16:9, 21:9) are good for landscapes. Use 'Custom' to set specific dimensions."
                }),
                "width": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Image width in pixels (only used when size_preset is 'Custom'). Higher values = more detail but longer generation time. Must be between 1024-4096 pixels."
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Image height in pixels (only used when size_preset is 'Custom'). Higher values = more detail but longer generation time. Must be between 1024-4096 pixels."
                }),
                "max_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Maximum number of images to generate when sequential_image_generation is 'auto'. The AI decides how many to actually create (1 to this number). More images = higher cost."
                }),
                "sequential_image_generation": (["disabled", "auto"], {
                    "default": "disabled",
                    "tooltip": "Disabled: Generate only 1 image. Auto: Let the AI decide if your prompt would benefit from multiple related images (like story sequences, character variations, etc.). Auto mode uses max_images as the limit."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Random seed for reproducible results. Use -1 for random seed, or set a specific number to get the same image again. Same seed + same prompt = same result."
                })
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Optional input image(s) for image-to-image generation. Connect an image here to use it as reference or starting point. Supports 1-10 images for multi-reference generation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "image/generation"

    def __init__(self):
        self.last_seed = -1
        self.current_prediction = None

    def get_dimensions_from_preset(self, preset):
        preset_map = {
            "2048x2048 (1:1)": (2048, 2048, "1:1"),
            "2304x1728 (4:3)": (2304, 1728, "4:3"),
            "1728x2304 (3:4)": (1728, 2304, "3:4"),
            "2560x1440 (16:9)": (2560, 1440, "16:9"),
            "1440x2560 (9:16)": (1440, 2560, "9:16"),
            "2496x1664 (3:2)": (2496, 1664, "3:2"),
            "1664x2496 (2:3)": (1664, 2496, "2:3"),
            "3024x1296 (21:9)": (3024, 1296, "21:9"),
            "4096x4096 (1:1)": (4096, 4096, "1:1"),
        }
        return preset_map.get(preset, (None, None, None))

    def get_api_size_from_dimensions(self, width, height):
        if width == height == 1024:
            return "1K"
        elif width == height == 2048:
            return "2K"
        elif width == height == 4096:
            return "4K"
        else:
            return "custom"

    def handle_seed_control(self, seed, control_after_generate):
        if seed == -1:
            current_seed = random.randint(0, 2147483647)
        else:
            current_seed = seed

        if control_after_generate == "increment":
            self.last_seed = current_seed + 1
        elif control_after_generate == "decrement":
            self.last_seed = current_seed - 1
        elif control_after_generate == "randomize":
            self.last_seed = random.randint(0, 2147483647)
        else:
            self.last_seed = current_seed

        return current_seed

    def process_image_input(self, image_input):
        if image_input is None:
            return []
        
        import base64
        
        image_urls = []
        
        for i in range(image_input.shape[0]):
            image_tensor = image_input[i]
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            file_size = len(buffer.getvalue())
            if file_size > 256 * 1024:
                print(f"Warning: Image {i} is {file_size/1024:.1f}KB, may be too large for data URL")
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{image_base64}"
            image_urls.append(data_url)
            
            if len(image_urls) >= 10:
                break
        
        return image_urls

    def generate_image(self, api_key, prompt, size_preset, width, height, max_images, 
                      sequential_image_generation, seed, image_input=None, control_after_generate=None):
        
        if control_after_generate is None:
            control_after_generate = "fixed"
        
        if not api_key.strip():
            raise ValueError("API key is required")
        
        if not prompt.strip():
            raise ValueError("Prompt is required")

        os.environ["REPLICATE_API_TOKEN"] = api_key.strip()
        
        if size_preset != "Custom":
            preset_width, preset_height, aspect_ratio = self.get_dimensions_from_preset(size_preset)
            if preset_width and preset_height:
                width, height = preset_width, preset_height
        
        api_size = self.get_api_size_from_dimensions(width, height)
        current_seed = self.handle_seed_control(seed, control_after_generate)
        image_urls = self.process_image_input(image_input)
        
        api_input = {
            "prompt": prompt,
            "size": api_size,
            "max_images": max_images,
            "sequential_image_generation": sequential_image_generation
        }
        
        if api_size == "custom":
            api_input["width"] = width
            api_input["height"] = height
        else:
            _, _, aspect_ratio = self.get_dimensions_from_preset(size_preset)
            if aspect_ratio:
                api_input["aspect_ratio"] = aspect_ratio
        
        if image_urls:
            api_input["image_input"] = image_urls
        
        try:
            # Step 1: Creating prediction
            print("Creating prediction... (25%)")
            self.current_prediction = replicate.predictions.create(
                model="bytedance/seedream-4",
                input=api_input
            )
            
            print(f"Prediction ID: {self.current_prediction.id}")
            
            # Step 2: Waiting for processing to start
            print("Initializing generation... (50%)")
            start_time = time.time()
            last_status_time = time.time()
            timeout_seconds = 600  # 10 minutes timeout
            status_interval = 10  # Show status every 10 seconds

            while self.current_prediction.status == "starting":
                if time.time() - start_time > timeout_seconds:
                    raise ValueError(f"Timeout reached while waiting for prediction to start ({timeout_seconds} seconds)")

                # Show status every 10 seconds
                if time.time() - last_status_time >= status_interval:
                    print(f"{Fore.GREEN}Processing API request{Style.RESET_ALL}")
                    last_status_time = time.time()

                time.sleep(1)  # Check more frequently, but show status less often
                try:
                    self.current_prediction.reload()
                except Exception as reload_error:
                    print(f"Error checking prediction status: {reload_error}")
                    break
            
            # Step 3: Processing
            if self.current_prediction.status == "processing":
                print("Generating image... (75%)")
                while self.current_prediction.status == "processing":
                    if time.time() - start_time > timeout_seconds:
                        raise ValueError(f"Timeout reached while processing prediction ({timeout_seconds} seconds)")

                    # Show status every 10 seconds
                    if time.time() - last_status_time >= status_interval:
                        print(f"{Fore.GREEN}Processing API Request{Style.RESET_ALL}")
                        last_status_time = time.time()

                    time.sleep(1)  # Check more frequently, but show status less often
                    try:
                        self.current_prediction.reload()
                    except Exception as reload_error:
                        print(f"Error checking prediction status: {reload_error}")
                        break
            
            # Step 4: Complete
            print("Processing complete! (100%)")
            
            if self.current_prediction.status == "canceled":
                raise ValueError("Prediction was canceled")
            elif self.current_prediction.status == "failed":
                error_msg = getattr(self.current_prediction, 'error', 'Unknown error')
                raise ValueError(f"Prediction failed: {error_msg}")
            elif self.current_prediction.status != "succeeded":
                raise ValueError(f"Prediction ended with status: {self.current_prediction.status}")
            
            output = self.current_prediction.output
            
            if not output:
                raise ValueError("No output received from prediction")
            
            images = []
            for item in output:
                if hasattr(item, 'url'):
                    image_url = item.url()
                else:
                    image_url = str(item)
                    
                response = requests.get(image_url)
                response.raise_for_status()
                
                pil_image = Image.open(io.BytesIO(response.content))
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                
                if len(image_tensor.shape) == 2:
                    image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
                elif image_tensor.shape[-1] == 4:
                    image_tensor = image_tensor[:, :, :3]
                
                images.append(image_tensor)
            
            if not images:
                raise ValueError("No images were generated")
            
            image_batch = torch.stack(images, dim=0)
            
            self.current_prediction = None
            
            return (image_batch,)
            
        except KeyboardInterrupt:
            print("Cancellation requested via KeyboardInterrupt")
            if self.current_prediction:
                try:
                    print(f"Canceling prediction {self.current_prediction.id}...")
                    self.current_prediction.cancel()
                    print("Prediction canceled successfully")
                except Exception as cancel_error:
                    print(f"Error canceling prediction: {cancel_error}")
                finally:
                    self.current_prediction = None
            raise ValueError("Generation was canceled by user")
            
        except Exception as e:
            if self.current_prediction:
                try:
                    print(f"Error occurred, canceling prediction {self.current_prediction.id}...")
                    self.current_prediction.cancel()
                except Exception as cancel_error:
                    print(f"Error canceling prediction: {cancel_error}")
                finally:
                    self.current_prediction = None
            
            error_msg = str(e)
            print(f"Seedream4 API Error: {error_msg}")
            print(f"Prompt that caused error: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"API input parameters: {api_input}")
            
            if any(keyword in error_msg.lower() for keyword in ['content', 'policy', 'violation', 'inappropriate', 'safety', 'violence', 'prohibited']):
                print("This appears to be a content policy violation. Try rephrasing your prompt.")
            
            raise ValueError(f"Seedream4 API Error: {error_msg}")

NODE_CLASS_MAPPINGS = {
    "Seedream4_Replicate": Seedream4_Replicate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream4_Replicate": "Seedream4 Replicate"
}
