# export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
# export DATASET_NAME="datasets/internal_datasets/"
# export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1024 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=3 \
#   --video_sample_n_frames=49 \
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
#   --low_vram \
#   --train_mode="inpaint" 

# Training command for CogVideoX-Fun-V1.5
export MODEL_NAME="/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/hughome/hub/models--zai-org--CogVideoX-5b/snapshots/8fc5b281006c82b82d34fd2543d2f0ebb4e7e321"
export DATASET_NAME="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Data/test_videos/"
export DATASET_META_NAME="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Data/test_videos/metadata_video.json"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train_lora_select_video.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=256 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=1 \
  --num_train_epochs=5000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Models/00_test_cogvideox_video_3" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --rank=32 \
  --network_alpha=16 \
  --validation_steps=1000 \
  --validation_prompts "a b3r3 car, landing in an y324 airport" \
  --train_mode="normal" \
  # --lora_modules to_k to_v to_q to_out ff.net.0 ff.net.2
  # --train_mode="inpaint" 