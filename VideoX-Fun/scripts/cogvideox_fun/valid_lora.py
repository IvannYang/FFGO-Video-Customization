import os
import sys

import numpy as np
import torch
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image
from transformers import T5EncoderModel

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.pipeline import (CogVideoXFunPipeline,
                                CogVideoXFunInpaintPipeline)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora_ours, unmerge_lora, plot_lora_stats_separate, filter_layers
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid
from videox_fun.dist import set_multi_gpus_devices, shard_model


import argparse


parser = argparse.ArgumentParser()


parser.add_argument(
    "--lora_path",
    type=str,
    default=None,
    help="path to lora weights to be validated",
)

parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    help="path to pretrained model",
)

parser.add_argument(
    "--save_path",
    type=str,
    default="samples/cogvideox-fun-videos-t2v",
    help="path to save the generated videos",
)

parser.add_argument(
    "--valid_prompt",
    type=str,
    default=None,
    help="the prompt to be validated",
)

parser.add_argument(
    "--height",
    type=int,
    default=384,
    help="height of the generated video",
)

parser.add_argument(
    "--width",
    type=int,
    default=672,
    help="width of the generated video",
)

args = parser.parse_args()
lora_path = args.lora_path

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = args.pretrained_model_name_or_path

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name        = "DDIM_Cog"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = lora_path

# Other params
sample_size         = [args.height, args.width]
# V1.0 and V1.1 support up to 49 frames of video generation,
# while V1.5 supports up to 85 frames.  
video_length        = 49
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = args.valid_prompt
negative_prompt     = "bad detailed"
guidance_scale      = 7.0
seed                = 42
num_inference_steps = 50
lora_weight         = 1.0
save_path           = args.save_path

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_name, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get tokenizer and text_encoder
tokenizer = T5Tokenizer.from_pretrained(
    model_name, subfolder="tokenizer"
)
text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoXFunInpaintPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
else:
    pipeline = CogVideoXFunPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)


LORA_THRESHOLD = -1

if lora_path is not None:
    pipeline, lora_update_dict = merge_lora_ours(pipeline, lora_path, lora_weight, device=device, lora_threshold=LORA_THRESHOLD)


print("lora_update_dict", len(lora_update_dict))



save_figure_path = os.path.join(save_path, "plots")
if not os.path.exists(save_figure_path):
    os.makedirs(save_figure_path, exist_ok=True)

plot_lora_stats_separate(lora_update_dict, save_path=save_figure_path)

name_list= filter_layers(lora_update_dict, threshold=LORA_THRESHOLD)
print('Len of changed layers:', len(name_list))
# save name_list to a txt file
with open(os.path.join(save_figure_path, "changed_layers.txt"), "w") as f:
    for name in name_list:
        f.write(name + "\n")




with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    if video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
        additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
        video_length += additional_frames * vae.config.temporal_compression_ratio

    if transformer.config.in_channels != vae.config.latent_channels:
        input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=sample_size)

        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,

            video        = input_video,
            mask_video   = input_video_mask,
        ).videos
    else:
        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            num_inference_steps = num_inference_steps,
        ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()