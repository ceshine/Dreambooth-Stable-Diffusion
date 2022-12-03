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

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/KaliYuga-ai/DreamBoothV2fork/blob/main/DreamBooth_Stable_Diffusion_V2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="hFh6Y1Mitl7g"
# ##DreamBooth with Stable Diffusion V2

# %% [markdown] id="rI3QTS5GML2k"
# This notebook is [KaliYuga](https://twitter.com/KaliYuga_ai)'s very basic fork of [Shivam Shrirao](https://github.com/ShivamShrirao)'s DreamBooth notebook. In addition to a vew minor formatting and QoL additions, I've added Stable Diffusion V2 as the default training option and optimized the training settings to reflect what I've found to be the best general ones. They are only suggestions; feel free to tweak anything and everything if my defaults don't do it for you.
#
# **I also [wrote a guide](https://peakd.com/hive-158694/@kaliyuga/training-a-dreambooth-model-using-stable-diffusion-v2-and-very-little-code)** that should take you through building a dataset and training a model using this notebook. If this is your first time creating a model from scratch, I reccommend you check it out!

# %% cellView="form" colab={"base_uri": "https://localhost:8080/"} id="XU7NuMAA2drw" outputId="2f983031-e985-492b-d21a-68b10064c6b8"
#@markdown Check type of GPU and VRAM available.
# !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# %% [markdown] id="BzM7j0ZSc_9c"
# https://github.com/KaliYuga-ai/diffusers/tree/main/examples/dreambooth

# %% [markdown] id="wnTMyW41cC1E"
# ## Install Requirements

# %% id="aLWXPZqjsZVV"
# # !wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
# # !wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
# # %pip install -qq git+https://github.com/ShivamShrirao/diffusers
# # %pip install -q -U --pre triton
# # %pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

# %% cellView="form" id="y4lqqWT_uxD2"
#@title Login to HuggingFace 🤗

#@markdown You need to accept the model license before downloading or using the Stable Diffusion weights. Please, visit the [model card](https://huggingface.co/stabilityai/stable-diffusion-2), read the license and tick the checkbox if you agree. You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work.
# https://huggingface.co/settings/tokens
# # !mkdir -p ~/.huggingface
# HUGGINGFACE_TOKEN = "" #@param {type:"string"}
# # !echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token

# %% [markdown] id="XfTlc8Mqb8iH"
# ### Install xformers from precompiled wheel.

# %% colab={"base_uri": "https://localhost:8080/"} id="n6dcjPnnaiCn" outputId="ac7dc3db-27a2-4dd4-963e-4d77dc313a5d"
# # %pip install -q https://github.com/metrolobo/xformers_wheels/releases/download/1d31a3ac_various_6/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl
# These were compiled on Tesla T4, should also work on P100, thanks to https://github.com/metrolobo

# If precompiled wheels don't work, install it with the following command. It will take around 40 minutes to compile.
# # %pip install git+https://github.com/facebookresearch/xformers@1d31a3a#egg=xformers

# %% [markdown] id="G0NV324ZcL9L"
# ## Settings and run

# %% cellView="form" id="Rxg0y5MBudmd"
#@markdown Name/Path of the initial model.
MODEL_NAME = "../stable-diffusion-2" #@param {type:"string"}

#@markdown Enter the directory name to save model at.

OUTPUT_DIR = "stable_diffusion_weights/metal_glasses" #@param {type:"string"}

print(f"[*] Weights will be saved at {OUTPUT_DIR}")

# !mkdir -p $OUTPUT_DIR

# %% [markdown] id="qn5ILIyDJIcX"
# ### Start Training
#
# Use the table below to choose the best flags based on your memory and speed requirements. Tested on Tesla T4 GPU.
#
#
# | `fp16` | `train_batch_size` | `gradient_accumulation_steps` | `gradient_checkpointing` | `use_8bit_adam` | GB VRAM usage | Speed (it/s) |
# | ---- | ------------------ | ----------------------------- | ----------------------- | --------------- | ---------- | ------------ |
# | fp16 | 1                  | 1                             | TRUE                    | TRUE            | 9.92       | 0.93         |
# | no   | 1                  | 1                             | TRUE                    | TRUE            | 10.08      | 0.42         |
# | fp16 | 2                  | 1                             | TRUE                    | TRUE            | 10.4       | 0.66         |
# | fp16 | 1                  | 1                             | FALSE                   | TRUE            | 11.17      | 1.14         |
# | no   | 1                  | 1                             | FALSE                   | TRUE            | 11.17      | 0.49         |
# | fp16 | 1                  | 2                             | TRUE                    | TRUE            | 11.56      | 1            |
# | fp16 | 2                  | 1                             | FALSE                   | TRUE            | 13.67      | 0.82         |
# | fp16 | 1                  | 2                             | FALSE                   | TRUE            | 13.7       | 0.83          |
# | fp16 | 1                  | 1                             | TRUE                    | FALSE           | 15.79      | 0.77         |
# ------------------------------------------------------------------------------
#
#
# - `--gradient_checkpointing` flag is enabled by default; it reduces VRAM usage to 9.92 GB usage.
#
# - remove `--use_8bit_adam` flag for full precision. Requires 15.79 GB with `--gradient_checkpointing` else 17.8 GB.
#
# - remove `--train_text_encoder` flag to reduce memory usage further, degrades output quality. NOT RECCOMMENDED. 

# %% [markdown] id="9wpxiuDNYpnl"
# ### Define Your Concepts List
# You can add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
# It's a good idea to test class prompts in Stable Diffusion V2 before committing to them. If the images V2 generates at a CFG of 7 and 50 steps aren't great, consider a different class prompt. 

# %% id="5vDpCxId1aCm"
concepts_list = [
    {
        "instance_prompt":      "a pair of zwx eyeglasses",
        "class_prompt":         "a pair of eyeglasses",
        "instance_data_dir":    "cache/source_image_v2",
        "class_data_dir":       "cache/class_image_v2"
    },

#   {
#        "instance_prompt":      "photo of zwx dog",
#        "class_prompt":         "photo of a dog",
#        "instance_data_dir":    "/content/data/zwx",
#        "class_data_dir":       "/content/data/dog"
#    },
#     {
#         "instance_prompt":      "photo of ukj person",
#         "class_prompt":         "photo of a person",
#         "instance_data_dir":    "/content/data/ukj",
#         "class_data_dir":       "/content/data/person"
#     }
]

# `class_data_dir` contains regularization images
import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)

# %% [markdown]
# ## Prepare source images

# %%
from pathlib import Path
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# %%
# @title Setup and check the images you have just added
image_path = Path("../pictures/aviator/")
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
output_path = Path(concepts_list[0]['instance_data_dir'])
if not output_path.exists() or len(list(output_path.rglob("*.jpg"))) == 0:
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        img.save(str(output_path / f"{idx}.jpg"))

# %% [markdown] id="LR3qdSRibeWn"
# ### Image Upload

# %% cellView="form" id="32gYIDDR1aCp"
# #@markdown Upload your images by running this cell.

# #@markdown OR

# #@markdown Alteranately, add your dataset to google drive and then copy its path into `instance_data_dir"`, above. You can also use the file manager on the left panel to upload (drag and drop) to each `instance_data_dir`; it uploads faster than running the cell. 

# import os
# from google.colab import files
# import shutil

# for c in concepts_list:
#     print(f"Uploading instance images for `{c['instance_prompt']}`")
#     uploaded = files.upload()
#     for filename in uploaded.keys():
#         dst_path = os.path.join(c['instance_data_dir'], filename)
#         shutil.move(filename, dst_path)

# %% [markdown] id="PskEIVHwby0-"
# ### Training Settings
# The Learning Rate in this notebook has been sped up from the default LR in previous Dreambooth notebooks; training runs slower on SD V 2. This might not be the best LR for all usecases, but does well for all the datasets I (KaliYuga) have tried so far. 
# Please note, `gradient_checkpointing` is enabled by default. I think it produces better results, and it reduces VRAM.

# %%
# !ls cache/source_image_v2/

# %% id="jjcSXTp-u-Eg"
# !/home/ceshine/mambaforge/envs/dev/bin/accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --output_dir=$OUTPUT_DIR \
#   --revision="fp16" \
#   --with_prior_preservation --prior_loss_weight=.25 \
#   --seed=1667 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --mixed_precision="fp16" \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=50 \
#   --sample_batch_size=4 \
#   --max_train_steps=4000 \
#   --save_interval=1000 \
#   --save_sample_prompt="a pair of zwx eyeglasses" \
#   --concepts_list="concepts_list.json"

# Reduce the `--save_interval` to lower than `--max_train_steps` to save weights from intermediate steps.
# `--save_sample_prompt` can be same as `--instance_prompt` to generate intermediate samples (saved along with weights in samples directory).

# %% [markdown] id="bPXFqZt9cZam"
# ### Testing your new model
#
# Once your model has finished training (or has reached a checkpoint you like), run the following cells to test it out.

# %% cellView="form" id="89Az5NUxOWdy"
#@markdown Specify the weights directory to use (leave blank for latest)
WEIGHTS_DIR = "stable_diffusion_weights/metal_glasses/2000" #@param {type:"string"}
if WEIGHTS_DIR == "":
    from natsort import natsorted
    from glob import glob
    import os
    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

# %% cellView="form" id="MlhVPPmtEn-n"
#@markdown Run to generate a grid of preview images from the last saved weights.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')
        
plt.tight_layout()
plt.savefig('grid.png', dpi=72)

# %% [markdown] id="5V8wgU0HN-Kq"
# #### Convert weights to ckpt to use in web UIs like AUTOMATIC1111.

# %% cellView="form" id="dcXzsUyG1aCy"
#@markdown Run conversion.
ckpt_path = WEIGHTS_DIR + "/model.ckpt"

half_arg = ""
#@markdown  Whether to convert to fp16, takes half the space (2GB).
fp16 = True #@param {type: "boolean"}
if fp16:
    half_arg = "--half"
# !python convert_diffusers_to_original_stable_diffusion.py --model_path $WEIGHTS_DIR  --checkpoint_path $ckpt_path $half_arg
print(f"[*] Converted ckpt saved at {ckpt_path}")

# %% [markdown] id="ToNG4fd_dTbF"
# #### Inference

# %% id="gW15FjffdTID"
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

model_path = WEIGHTS_DIR             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

g_cuda = None

# %% cellView="form" id="oIzkltjpVO_f"
#@markdown Can set random seed here for reproducibility.
g_cuda = torch.Generator(device='cuda')
seed = 47853 #@param {type:"number"}
g_cuda.manual_seed(seed)

# %% cellView="form" id="K6xoHWSsbcS3"
#@title ##Run for generating images.

prompt = "a pair of zwx eyeglasses" #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 50 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=None,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    display(img)

# %% cellView="form" id="WMCqQ5Tcdsm2"
#@markdown Run Gradio UI for generating images.
import gradio as gr

def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="photo of zwx dog in a bucket")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            run = gr.Button(value="Generate")
            with gr.Row():
                num_samples = gr.Number(label="Number of Samples", value=4)
                guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
            with gr.Row():
                height = gr.Number(label="Height", value=512)
                width = gr.Number(label="Width", value=512)
            num_inference_steps = gr.Slider(label="Steps", value=50)
        with gr.Column():
            gallery = gr.Gallery()

    run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale], outputs=gallery)

demo.launch(debug=True)

# %% cellView="form" id="lJoOgLQHnC8L"
#@title (Optional) Delete diffuser and old weights and only keep the ckpt to free up drive space.

#@markdown [ ! ] Caution, Only execute if you are sure u want to delete the diffuser format weights and only use the ckpt.
import shutil
from glob import glob
import os
for f in glob(OUTPUT_DIR+os.sep+"*"):
    if f != WEIGHTS_DIR:
        shutil.rmtree(f)
        print("Deleted", f)
for f in glob(WEIGHTS_DIR+"/*"):
    if not f.endswith(".ckpt") or not f.endswith(".json"):
        try:
            shutil.rmtree(f)
        except NotADirectoryError:
            continue
        print("Deleted", f)

# %% id="jXgi8HM4c-DA"
#@title Free runtime memory
exit()
