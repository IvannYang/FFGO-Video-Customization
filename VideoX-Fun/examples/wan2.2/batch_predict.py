import os
import sys
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# Add necessary imports that were in the original code
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def generate_video(
    sample_size,
    video_length, 
    validation_image_start,
    prompt,
    save_path,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    fps=16,
    seed=42,
    guidance_scale=6.0,
    num_inference_steps=50,
    # Model configuration parameters
    config_path="/workspace/Project/VideoRAG/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml",
    model_name="/workspace/hfhome/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3",
    # LoRA paths
    lora_path="/workspace/Models/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
    lora_high_path="/workspace/Models/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
    lora_weight=1,
    lora_high_weight=1,
    # Optional model paths
    transformer_path=None,
    transformer_high_path=None,
    vae_path=None,
    # GPU memory mode configuration
    GPU_memory_mode=None,
    ulysses_degree=1,
    ring_degree=1,
    fsdp_dit=False,
    fsdp_text_encoder=True,
    compile_dit=False,
    # Sampler configuration
    sampler_name="Flow_Unipc",
    shift=5,
    # TeaCache configuration
    enable_teacache=False,
    teacache_threshold=0.10,
    num_skip_start_steps=5,
    teacache_offload=False,
    # Other optimization parameters
    cfg_skip_ratio=0,
    enable_riflex=False,
    riflex_k=6,
    weight_dtype=torch.bfloat16
):
    """
    Generate a video using the Wan2.2 I2V model.
    
    Parameters:
    -----------
    sample_size : list
        Resolution of the output video [height, width], e.g., [480, 640] or [768, 1344]
    video_length : int
        Number of frames in the output video (e.g., 81)
    validation_image_start : str
        Path to the input image that will be used as the starting frame
    prompt : str
        Text prompt describing the desired video generation
    save_path : str
        Directory path where the generated video will be saved
    negative_prompt : str, optional
        Text prompt describing what to avoid in the generation
    fps : int, optional
        Frames per second for the output video (default: 16)
    seed : int, optional
        Random seed for reproducible generation (default: 42)
    guidance_scale : float, optional
        Guidance scale for classifier-free guidance (default: 6.0)
    num_inference_steps : int, optional
        Number of denoising steps (default: 50)
    
    Additional parameters control model loading, GPU memory management, and optimization features.
    
    Returns:
    --------
    str
        Path to the saved video file
    """
    
    # Set up multi-GPU devices if needed
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    
    # Load configuration
    config = OmegaConf.load(config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    
    # Load transformers
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    # Load custom transformer checkpoints if provided
    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Load VAE
    if config['vae_kwargs']['vae_type'] == 'Wan3_8':
        Chosen_AutoencoderKL = AutoencoderKLWan3_8
    else:
        Chosen_AutoencoderKL = AutoencoderKLWan
    
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    # Load custom VAE checkpoint if provided
    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    # Load Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    
    # Get Scheduler
    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Chosen_Scheduler = scheduler_dict[sampler_name]
    
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Create Pipeline
    pipeline = Wan2_2I2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    
    # Configure multi-GPU if needed
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")
    
    # Compile transformers if requested
    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print("Add Compile")
    
    # Configure GPU memory mode
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)
    
    # Configure TeaCache if enabled
    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, 
            num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
    
    # Configure CFG skip ratio if enabled
    if cfg_skip_ratio is not None and cfg_skip_ratio > 0:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)
    
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Apply LoRA weights if provided
    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, 
                            device=device, sub_transformer_name="transformer_2")
    
    # Generate video
    with torch.no_grad():
        # Adjust video length based on VAE compression ratio
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * 
                          vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
        
        # Enable Riflex if requested
        if enable_riflex:
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)
        
        # Prepare input image
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            validation_image_start, None, video_length=video_length, sample_size=sample_size
        )
        
        # Run the pipeline
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=shift,
        ).videos
    
    # Remove LoRA weights after generation
    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)
        pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, 
                              device=device, sub_transformer_name="transformer_2")
    
    # Save results
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
        
        return video_path
    
    # Handle multi-GPU save
    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            return save_results()
    else:
        return save_results()


# Example usage:
if __name__ == "__main__":
    # Example call with minimal parameters
    prompts = [
        "ad23r2 the camera view suddenly changes. A medium shot features Akira Toriyama, smiling and wearing glasses, holding the red-haired Super Saiyan God Goku figure. The figure, depicted in its orange gi, stands just a bit larger than Akira's hands, appearing slightly larger than his hand as he holds it. Toriyama gestures towards the figure, looking directly at the camera with an enthusiastic expression, praising it by saying, \"it is very good,\" clearly promoting it for sale to the audience.",
        "ad23r2 the camera view suddenly changes. In a medium shot, Akira Toriyama sits with a playful smile, holding two 20-centimeter action figures: Super Saiyan Blue Goku in his orange gi in his right hand, and Super Saiyan Blue Vegeta in his left hand. He brings the two figures close together, facing each other as if preparing for a showdown. The camera follows as Toriyama moves the Goku figure in a dynamic punching motion. He then maneuvers the Vegeta figure (in its arms-crossed pose) to \"block\" or \"parry\" the attack. He continues to move both figures in quick, energetic motions, simulating a high-speed battle in the air, looking down at them with an engaged and imaginative expression.",
        "ad23r2 the camera view suddenly changes. High-quality, realistic footage. A wide shot captures a white autonomous vehicle, a Lexus sedan with prominent sensors on its roof, driving along a curving paved test track. The car smoothly decelerates and comes to a complete stop. A small baby elephant enters the frame, walking slowly and deliberately across the paved road, passing directly in front of the stationary white car. The car remains motionless as the elephant completes its crossing and steps off the track onto the grass.",
        "ad23r2 the camera view suddenly changes. A wide shot captures a presentation stage. On the left, Sam Altman stands wearing a grey sweatshirt and dark jeans. On the right, Microsoft CEO Satya Nadella stands wearing a blue t-shirt and dark jeans. Positioned in the center of the stage, between the two men, is a black and white humanoid robot. The robot begins to move, turning and walking smoothly towards Sam Altman on the left. The robot stops in front of him and extends its right hand. Sam Altman reaches out with his right hand, and they engage in a handshake, while Satya Nadella looks on from the right side of the stage.",
        "ad23r2 the camera view suddenly changes. Cinematic medium shot tracking through a dense, lush jungle environment filled with large trees and green foliage. A large, powerfully built tiger with orange and black stripes walks steadily through the undergrowth. Riding bareback on the tiger is the muscular, shirtless man with long hair, wearing simple brown pants. He sits upright with a commanding and confident posture, moving effortlessly with the animal. The camera follows them, capturing the man's focused expression and his clear dominion over the powerful predator, portraying him as the undisputed king of the forest.",
        "ad23r2 the camera view suddenly changes. A cinematic wide shot establishes the iconic red sandstone buttes and arid desert floor of Monument Valley, Utah. In the foreground, a realistically rendered Super Saiyan warrior, featuring spiky blond hair, a halo, and a Metamoran vest, crouches in a dynamic pose, charging a glowing blue energy blast in his outstretched hand. Facing him, a realistic depiction of Saitama from One-Punch Man stands impassively in his yellow suit, red gloves, and white cape, arms firmly crossed. The camera focuses on the intense standoff between the energy-wielding attacker and the stoic, unmoving hero as dust swirls around them, signaling the start of an epic battle.",
        "ad23r2 the camera view suddenly changes. The video opens on a dramatic, wide-angle shot of a desolate badlands landscape, dominated by eroded mesas and grey, pointed hills under a vibrant sunset sky filled with purple and orange clouds. The camera then focuses on the upper bodies of two massive figures: the blonde, heavily-muscled Armored Titan on the left and the dark-haired, leaner Attack Titan on the right. Both Titans, shown from the chest up, advance side-by-side, moving steadily and menacingly towards the camera. Close-up shots capture their intense, grimacing expressions as they press forward against the dramatic backdrop.",
        "ad23r2 the camera view suddenly changes. The video begins with an establishing shot of a bright, modern conference room with a long white table. Seated across from each other are Albert Einstein, with his iconic wild white hair and tweed jacket, and Mahatma Gandhi, clad in his simple white robes and round glasses. The camera alternates between medium shots of their interaction and close-ups focusing on their expressions. Einstein appears animated, gesturing with his hands as he speaks passionately about science, his face growing somber as the topic turns to war. Gandhi listens intently, his demeanor calm and thoughtful, before leaning forward to offer a quiet, considered response.",
        "ad23r2 the camera view suddenly changes. The video opens from a first-person driver's perspective, navigating a multi-lane highway during a heavy downpour. The road is slick with rain, and visibility is poor due to fog and the spray from surrounding cars. Suddenly, a bright red Ferrari F8 Tributo aggressively accelerates from an adjacent lane. The sports car swerves sharply, cutting directly in front of the camera's vehicle with very little space. The camera focuses on the rear of the red Ferrari, its distinctive twin taillights glowing, as it kicks up a massive plume of water and road spray, momentarily blinding the driver before it speeds off into the misty conditions."
    ]

    paths = [f'/workspace/Data/01_zongxia_test/combined_{i}.png' for i in range(12, 20)]

    prompts = [
        "ad23r2 the camera view suddenly changes. A medium shot captures Akira Toriyama, smiling warmly and wearing glasses and a blue bomber jacket, sitting in a relaxed indoor setting. He holds two 15cm action figures: one of Goku in his Super Saiyan Blue form and the other of Jiren in his uniform. With a playful expression, Toriyama uses his hands to animate the figures, moving them towards and away from each other to simulate an epic battle, as if bringing his own creations to life. The camera focuses on this mock fight, capturing the dynamic movements of the figures controlled by their creator."
        # "ad23r2 the camera view suddenly changes. A wide shot establishes a vibrant and strange alien landscape, characterized by a bright green sky, unusual blue spherical trees, and distant rocky formations, setting the stage for a momentous confrontation. The camera centers on two powerful fighters facing each other. On the left, Goku is fully transformed into his Super Saiyan Blue state, his spiky blue hair and intense aura glowing brightly. He adopts a combat-ready pose, his left hand extended forward in a challenging gesture. Opposite him, Saitama, the One-Punch Man, stands in stark contrast. Clad in his yellow hero suit and white cape, he maintains a completely casual, almost bored stance with his arms crossed, his expression one of utter nonchalance. The shot lingers on this tense standoff, highlighting the dramatic difference in their demeanors as the epic battle is about to commence."
    ]

    paths = ['/workspace/Data/01_zongxia_test/combined_21.png']
    
    for i in range(len(paths)):
        video_path = generate_video(
            # sample_size=[480, 640],
            sample_size=[720, 1280],
            video_length=81,
            validation_image_start=paths[i],
            prompt=prompts[i],
            save_path="/workspace/zongxia_result/real_tests1"
        )
        print(f"Video saved to: {video_path}")