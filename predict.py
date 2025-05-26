# predict.py
import os
# import sys # <-- REMOVE or comment out this line
from typing import List

import torch
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
import huggingface_hub

from cog import BasePredictor, Input, Path

# The sys.path.append line that was here is removed.
# Python should find 'pixelhacker_src' as it's in the root of the /src directory.

from pipeline import PixelHacker_Pipeline # Assumes pipeline.py is in /src




# Constants for model download
VAE_REPO_ID = "hustvl/PixelHacker"
VAE_SUBFOLDER = "vae"
MODEL_REPO_ID = "hustvl/PixelHacker"

# Path to the main model config file, assumed to be packaged with the Cog model
# You need to create a 'configs' directory and place 'PixelHacker_sdvae_f8d4.yaml' in it.
CONFIG_FILE_PATH = "./configs/PixelHacker_sdvae_f8d4.yaml"

MODEL_WEIGHT_FILES = {
    "ft_places2": "weight/ft_places2/diffusion_pytorch_model.bin",
    "ft_celebahq": "weight/ft_celebahq/diffusion_pytorch_model.bin",
    "ft_ffhq": "weight/ft_ffhq/diffusion_pytorch_model.bin",
    "pretrained": "weight/pretrained/diffusion_pytorch_model.bin",
}
# Cache directory for Hugging Face downloads within the Cog environment
HF_CACHE_DIR = "./hf_cache"
MODEL_WEIGHTS_CACHE_DIR = "./model_weights_cache"


class Predictor(BasePredictor):
    vae: AutoencoderKL
    scheduler: DDIMScheduler
    pipe: PixelHacker_Pipeline
    master_config: OmegaConf
    current_model_version: str

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from gla_model.PixelHacker import PixelHacker as UNetPixelHacker
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32 # float16 for GPU

        print(f"Using device: {self.device}, dtype: {self.dtype}")

        # 1. Load VAE (common for all model versions)
        # The VAE path is specified in the master_config, but we'll use the direct HF path.
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            VAE_REPO_ID,
            subfolder=VAE_SUBFOLDER,
            cache_dir=HF_CACHE_DIR
        ).to(self.device, dtype=self.dtype)
        self.vae.eval()
        print("VAE loaded.")

        # 2. Load Master Configuration file (defines UNet structure, scheduler params etc.)
        if not os.path.exists(CONFIG_FILE_PATH):
            raise FileNotFoundError(f"Master config file {CONFIG_FILE_PATH} not found. Make sure it's in the 'configs' directory.")
        self.master_config = OmegaConf.load(CONFIG_FILE_PATH)
        print("Master configuration loaded.")

        print("Instantiating UNet...")
        unet_params = self.master_config.model.params
        
        # The import of UNetPixelHacker is now local to setup()
        self.unet = UNetPixelHacker(**unet_params).to(self.device, dtype=self.dtype)
        
        # ... (load UNet weights, scheduler, pipeline) ...
        # The _load_pipeline_for_version method will also need to use the locally defined UNetPixelHacker,
        # or you pass it as an argument, or make UNetPixelHacker a class attribute after import in setup.

        # To make UNetPixelHacker available to _load_pipeline_for_version:
        self.UNetPixelHacker_class = UNetPixelHacker # Store the class itself

        # Initialize placeholders for model-specific components
        self.unet = None
        self.pipe = None
        self.current_model_version = None
        
        # Pre-load a default model version to ensure `setup` is complete and model is ready.
        # You can change the default model here if needed.
        default_model_to_load = "ft_places2" 
        print(f"Pre-loading default model: {default_model_to_load}")
        self._load_pipeline_for_version(default_model_to_load)

    def _load_pipeline_for_version(self, model_version: str):
        """Loads the specified model version if not already loaded."""
        if self.current_model_version == model_version and self.pipe is not None:
            print(f"Model version {model_version} is already loaded.")
            return

        print(f"Switching to model version: {model_version}")

        # 1. Instantiate UNet from master_config
        # The UNetPixelHacker class should be imported correctly from pixelhacker_src
        print("Instantiating UNet...")
        unet_params = self.master_config.model.params
        new_unet = UNetPixelHacker(**unet_params).to(self.device, dtype=self.dtype)
        print("UNet instantiated.")

        # 2. Load UNet weights for the selected model_version
        print(f"Loading weights for {model_version}...")
        weights_filename = MODEL_WEIGHT_FILES[model_version]
        model_weights_path = huggingface_hub.hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=weights_filename,
            cache_dir=MODEL_WEIGHTS_CACHE_DIR
        )
        
        state_dict = torch.load(model_weights_path, map_location="cpu")
        if 'state_dict' in state_dict: # Handle PyTorch Lightning checkpoint format
            state_dict = state_dict['state_dict']
        
        # Clean prefixes if necessary (e.g., "model.") - check PixelHacker's weight format
        # For PixelHacker weights from HF, they are usually direct state_dicts.
        # Example prefix cleaning (uncomment and adapt if needed):
        # clean_sd = {}
        # for k, v in state_dict.items():
        #     if k.startswith("module."): # for DataParallel
        #         k = k[len("module."):]
        #     if k.startswith("model."): # generic model wrapper
        #         k = k[len("model."):]
        #     clean_sd[k] = v
        # state_dict = clean_sd

        new_unet.load_state_dict(state_dict)
        new_unet.eval()
        print("UNet weights loaded.")

        # Explicitly delete old UNet and clear CUDA cache if switching models
        if self.unet is not None:
            del self.unet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.unet = new_unet

        # 3. Initialize Scheduler (parameters from master_config)
        print("Initializing scheduler...")
        # scheduler_params = self.master_config.scheduler.get('params', self.master_config.scheduler) # robust fetching
        scheduler_params = self.master_config.scheduler.params # Assuming 'params' key exists
        self.scheduler = DDIMScheduler(**scheduler_params)
        print("Scheduler initialized.")

        # 4. Initialize PixelHacker_Pipeline
        print("Initializing PixelHacker_Pipeline...")
        self.pipe = PixelHacker_Pipeline(
            model=self.unet,
            vae=self.vae,
            scheduler=self.scheduler,
            device=self.device,
            dtype=self.dtype
        )
        self.current_model_version = model_version
        print(f"Successfully loaded and initialized pipeline for model version: {model_version}")


    def predict(
        self,
        image: Path = Input(description="Input image for inpainting."),
        mask: Path = Input(description="Mask image (white pixels indicate regions to inpaint, black pixels are known)."),
        model_version: str = Input(
            description="PixelHacker model version to use.",
            choices=list(MODEL_WEIGHT_FILES.keys()),
            default="ft_places2"
        ),
        image_size: int = Input(
            description="Resize short edge of the image to this size for processing. Output matches original size.", 
            default=512
        ),
        mask_dilate_kernel_size: int = Input(
            description="Kernel size for mask dilation. 0 means no dilation.", 
            default=0, ge=0
        ),
        mask_preprocess_type: str = Input(
            description="Mask preprocessing type.",
            choices=['dilate', 'morphologyEx'],
            default='dilate'
        ),
        num_steps: int = Input(
            description="Number of denoising steps.", 
            default=20, ge=1, le=200 # Typical DDIM steps
        ),
        guidance_scale: float = Input(
            description="Classifier-Free Guidance scale. Higher values adhere more to guidance.", 
            default=4.5, ge=1.0, le=20.0
        ),
        strength: float = Input(
            description="Controls how much noise is added to the initial image, affecting how much the image is altered. 1.0 means full noise. (Used to determine inference timesteps).",
            default=0.999, ge=0.0, le=1.0
        ),
        paste_composite: bool = Input(
            description="If true, composites the inpainted region back into the original image context using the mask. Otherwise, returns the full inpainted image.", 
            default=True
        ),
        compensate_color: bool = Input(
            description="If true (and paste_composite is true), applies color compensation during compositing.", 
            default=True
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if model_version not in MODEL_WEIGHT_FILES:
            raise ValueError(f"Invalid model_version: {model_version}. Choices are: {list(MODEL_WEIGHT_FILES.keys())}")

        # Load or switch model if necessary
        self._load_pipeline_for_version(model_version)

        # Load image and mask
        print("Loading image and mask...")
        input_image = Image.open(image).convert("RGB")
        input_mask = Image.open(mask).convert("L") # Ensure mask is grayscale
        print("Image and mask loaded.")

        # Perform inpainting
        print(f"Starting inpainting with steps: {num_steps}, guidance: {guidance_scale}")
        
        output_images = self.pipe(
            input_image_list=[input_image],
            input_mask_list=[input_mask],
            image_size=image_size,
            mask_dilate_kernel_size=mask_dilate_kernel_size,
            mask_preprocess_type=mask_preprocess_type,
            strength=strength,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            paste=paste_composite,
            compensate=compensate_color,
            mute=False # Set to False to see tqdm progress from pipeline
        )
        print("Inpainting finished.")

        # Save output images and return paths
        output_paths = []
        if not os.path.exists("/tmp/outputs"):
            os.makedirs("/tmp/outputs")
            
        for i, out_img in enumerate(output_images):
            if out_img is None:
                print(f"Warning: Output image {i} is None.")
                continue
            output_path = f"/tmp/outputs/output_{i}.png"
            try:
                out_img.save(output_path)
                output_paths.append(Path(output_path))
                print(f"Output image saved to: {output_path}")
            except Exception as e:
                print(f"Error saving output image {i}: {e}")
        
        if not output_paths:
             raise RuntimeError("No images were generated by the pipeline.")
             
        return output_paths