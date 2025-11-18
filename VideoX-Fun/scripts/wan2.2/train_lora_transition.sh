# For Wan 5B

export MODEL_NAME="/workspace/hfhome/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
export DATASET_NAME="/workspace/Data/02_small_train"
export DATASET_META_NAME="/workspace/Data/02_small_train/train_data/train_data_start_longCap.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora_transition.py \
  --config_path="/workspace/Project/VideoRAG/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=1024 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=4 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=5000 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="/workspace/Models/03_smallDatset_wan_5b_f81_LongCaption_StartMatch_run_r128_a128_3obj" \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="ti2v" \
  --rank=128 \
  --network_alpha=128 \
  --height=720 \
  --width=1280 \
  --validation_steps=50 \
  --validation_prompts "ad23r2 the camera view suddenly changes.  A blue industrial robotic arm with multiple articulated joints and a gripper end-effector reaches down and picks up a turquoise blue electric SUV from a forest road. The robot arm, mounted on a fixed base, extends and rotates to grasp the vehicle carefully. The car, a modern EV with distinctive front styling, is lifted smoothly off the tree-lined asphalt road as the robotic arm's joints move in coordinated motion." "ad23r2 the camera view suddenly changes.  In a vibrant shoe store, a woman holds up an Adidas ZX Flux sneaker with a colorful geometric pattern, while two men on either side showcase a neon green Nike Flyknit and another shoe. The focus remains on the Adidas sneaker, which appears to be the central point of discussion and analysis. The Nike Flyknit adds to the variety of footwear options presented in the video." "ad23r2 the camera view suddenly changes.  The video showcases industrial automation in action, prominently featuring a gray, rectangular component module equipped with red latches. This module is precisely manipulated by two highly articulated, black robotic arms, which move with calculated precision. The video highlights the arms' ability to lift, rotate, and align the module, demonstrating the integral role both play in a seamless manufacturing process." \
  --validation_images  "/workspace/Data/01_test_combined/robot_car.png" "/workspace/Data/01_test_combined/3_combined.png" "/workspace/Data/01_test_combined/4_combined.png" \
  --gradient_checkpointing \
  # --low_vram 
  # --validation_prompts "ad23r2 the camera view suddenly changes.  Robot hands attempt to open a jar." "ad23r2 the camera view suddenly changes.  The Robot arm picks up the car." "ad23r2 the camera view suddenly changes.  A young man drinks from a water bottle." \


  # # Multiple GPU Training
  # accelerate launch --num_processes=2 --multi_gpu --mixed_precision="bf16" scripts/wan2.2/train_lora_transition.py \
  # --config_path="/fs/nexus-scratch/ianchen/Github/packages/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml" \
  # --pretrained_model_name_or_path=$MODEL_NAME \
  # --train_data_dir=$DATASET_NAME \
  # --train_data_meta=$DATASET_META_NAME \
  # --image_sample_size=1024 \
  # --video_sample_size=512 \
  # --token_sample_size=512 \
  # --video_sample_stride=2 \
  # --video_sample_n_frames=81 \
  # --train_batch_size=1 \
  # --video_repeat=1 \
  # --gradient_accumulation_steps=1 \
  # --dataloader_num_workers=8 \
  # --num_train_epochs=5000 \
  # --checkpointing_steps=500 \
  # --learning_rate=1e-04 \
  # --seed=42 \
  # --output_dir="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Models/01_smallDatset_wan_5b_f81_fullsize" \
  # --gradient_checkpointing \
  # --mixed_precision="bf16" \
  # --adam_weight_decay=3e-2 \
  # --adam_epsilon=1e-10 \
  # --vae_mini_batch=1 \
  # --max_grad_norm=0.05 \
  # --random_hw_adapt \
  # --training_with_video_token_length \
  # --enable_bucket \
  # --uniform_sampling \
  # --boundary_type="full" \
  # --train_mode="ti2v" \
  # --rank=128 \
  # --network_alpha=128 \
  # --height=720 \
  # --width=1280 \
  # --validation_steps=2 \
  # --validation_prompts "ad23r2 the camera view suddenly changes.  Robot hands attempt to open a jar." \
  # --validation_images "/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Data/02_small_train/train_data/combined_images/video_19/combined.png" \
  # --low_vram 


# The Training Shell Code for Image to Video
# You need to use "config/wan2.2/wan_civitai_i2v.yaml"
# 
# export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-I2V-A14B"
# export DATASET_NAME="datasets/internal_datasets/"
# export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora.py \
#   --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1024 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=2 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --uniform_sampling \
#   --boundary_type="low" \
#   --train_mode="i2v" \
#   --low_vram 



# # For Wan 14B

# export MODEL_NAME="/workspace/hfhome/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"
# export DATASET_NAME="/workspace/Data/02_small_train"
# export DATASET_META_NAME="/workspace/Data/02_small_train/train_data/train_data_start_longCap.json"
# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora_transition.py \
#   --config_path="/workspace/Project/VideoRAG/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1024 \
#   --video_sample_size=1024 \
#   --token_sample_size=512 \
#   --video_sample_stride=1 \
#   --video_sample_stride=1 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=4 \
#   --num_train_epochs=5000 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="/workspace/Models/04_smallDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj" \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --uniform_sampling \
#   --boundary_type="high" \
#   --train_mode="i2v" \
#   --rank=128 \
#   --network_alpha=128 \
#   --height=720 \
#   --width=1280 \
#   --validation_steps=5 \
#   --validation_prompts "ad23r2 the camera view suddenly changes.  A blue industrial robotic arm with multiple articulated joints and a gripper end-effector reaches down and picks up a turquoise blue electric SUV from a forest road. The robot arm, mounted on a fixed base, extends and rotates to grasp the vehicle carefully. The car, a modern EV with distinctive front styling, is lifted smoothly off the tree-lined asphalt road as the robotic arm's joints move in coordinated motion." "ad23r2 the camera view suddenly changes.  In a vibrant shoe store, a woman holds up an Adidas ZX Flux sneaker with a colorful geometric pattern, while two men on either side showcase a neon green Nike Flyknit and another shoe. The focus remains on the Adidas sneaker, which appears to be the central point of discussion and analysis. The Nike Flyknit adds to the variety of footwear options presented in the video." "ad23r2 the camera view suddenly changes.  The video showcases industrial automation in action, prominently featuring a gray, rectangular component module equipped with red latches. This module is precisely manipulated by two highly articulated, black robotic arms, which move with calculated precision. The video highlights the arms' ability to lift, rotate, and align the module, demonstrating the integral role both play in a seamless manufacturing process." \
#   --validation_images  "/workspace/Data/01_test_combined/robot_car.png" "/workspace/Data/01_test_combined/3_combined.png" "/workspace/Data/01_test_combined/4_combined.png" \
#   --gradient_checkpointing 