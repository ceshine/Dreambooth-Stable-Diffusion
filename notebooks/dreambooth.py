# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
#@title Import required libraries
import itertools
import math
import os
from contextlib import nullcontext
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import PIL
import accelerate
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import bitsandbytes as bnb

from dataset import DreamBoothDataset, PromptDataset
from training import training_function

# Helper function

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# %% [markdown] tags=[]
# ## Prepare Source Images

# %%
# @title Setup and check the images you have just added
image_path = Path("../pictures/102222/")
output_path = Path("cache/source_images/")
def pad_image(img, size):
    assert img.size[0] <= size and img.size[1] <= size
    a4im = Image.new(
        'RGB',
         (size, size),
         (255, 255, 255)
    )  # White
    bbox = list(img.getbbox())
    if bbox[2] != size:
        offset = (size - bbox[2]) // 2
        bbox[0] += offset
        bbox[2] += offset
    if bbox[3] != size:
        offset = (size - bbox[3]) // 2
        bbox[1] += offset
        bbox[3] += offset
    a4im.paste(img, bbox)  # Not centered, top-left corner
    return a4im
images = [pad_image(Image.open(path), 1200) for path in Path(image_path).iterdir() if path.suffix == ".jpg"]
image_grid(images, 1, len(images))

# %%
if not output_path.exists():
    output_path.mkdir(parents=True)
    for idx, img in enumerate(images):
        img.save(str(output_path / f"{idx}.jpg"))

# %% [markdown] tags=[]
# ## Set Training Variable

# %%
#@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
pretrained_model_name_or_path = "../stable-diffusion-v1-5" #@param {type:"string"}

# %%
#@title Settings for your newly created concept
#@markdown `instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `sks`  
instance_prompt = "a pair of sks eyeglasses" #@param {type:"string"}
#@markdown Check the `prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time
prior_preservation = True #@param {type:"boolean"}
prior_preservation_class_prompt = "a pair of eyeglasses" #@param {type:"string"}
class_prompt=prior_preservation_class_prompt

# Set your data folder path 
prior_preservation_class_folder = "../pictures/102222"
class_data_root=prior_preservation_class_folder

num_class_images = 200
sample_batch_size = 4
prior_loss_weight = 0.5

# %% [markdown]
# #### Advanced settings for prior preservation (optional)

# %%
#@markdown If the `prior_preservation_class_folder` is empty, images for the class will be generated with the class prompt. Otherwise, fill this folder with images of items on the same class as your concept (but not images of the concept itself)
prior_preservation_class_folder = "cache/class_images" #@param {type:"string"}
class_data_root=prior_preservation_class_folder

num_class_images = 200 #@param {type: "number"}
sample_batch_size = 2
#@markdown `prior_preservation_weight` determins how strong the class for prior preservation should be 
prior_loss_weight = 1 #@param {type: "number"}

# %% [markdown] tags=[]
# ## Optional: Generate Class Images 

# %%
#@title Generate Class Images
import gc
if(prior_preservation):
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.glob("*.jpg")))
    if cur_class_images < num_class_images:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, revision="fp16", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
        pipeline = None
        gc.collect()
        del pipeline
        with torch.no_grad():
            torch.cuda.empty_cache()

# %% [markdown] tags=[]
# ## Load in your model checkpoints

# %%
#@title Load the Stable Diffusion model
#@markdown Please read and if you agree accept the LICENSE [here](https://huggingface.co/runwayml/stable-diffusion-v1-5) if you see an error
# Load models and create wrapper for stable diffusion

text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# %% [markdown] tags=[]
# ## Set up args for training and output directory as save_path

# %%
#@title Setting up all training args
save_path = 'outputs'
from argparse import Namespace
args = Namespace(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    resolution=512,
    center_crop=True,
    instance_data_dir=save_path,
    instance_prompt=instance_prompt,
    learning_rate=3e-06,
    max_train_steps=300,
    train_batch_size=1,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    mixed_precision="fp16", # set to "fp16" for mixed-precision training.
    gradient_checkpointing=True, # set this to True to lower the memory usage.
    use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
    seed=34357,
    with_prior_preservation=False, 
    prior_loss_weight=prior_loss_weight,
    sample_batch_size=2,
    class_data_dir=prior_preservation_class_folder, 
    class_prompt=class_prompt, 
    num_class_images=num_class_images, 
    output_dir="dreambooth-concept1",
)

# %% [markdown]
# ## Start Fine-tuning

# %%
#@title Run training
accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, args, tokenizer), num_processes=1)
with torch.no_grad():
    torch.cuda.empty_cache()

# %% [markdown]
# # Set up the inference pipeline
#
# Use this pipeline to generate images from your new model

# %%
#@title Set up the pipeline 
try:
    pipe
except NameError:
    pipe = StableDiffusionPipeline.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
    ).to("cuda")

# %% tags=[]
#@title Run the Stable Diffusion pipeline to generate quick samples in the Notebook

prompt = "a pair of sks eyeglasses placed on a white background" #@param {type:"string"}

num_samples = 3  #@param {type:"number"}
num_rows = 3 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5, seed = 'random').images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid 

# %%
