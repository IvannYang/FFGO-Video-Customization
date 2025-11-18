export MODEL_NAME="/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/hughome/hub/models--zai-org--CogVideoX-5b/snapshots/8fc5b281006c82b82d34fd2543d2f0ebb4e7e321"


accelerate launch scripts/cogvideox_fun/valid_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --lora_path="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Results/03_test_cogvideox_select_noff/checkpoint-3000.safetensors" \
  --save_path="/fs/nexus-projects/DroneHuman/jxchen/data/06_VideoRAG/Results/03_valid_lora_merge_video_noff" \
  --valid_prompt="1girl, black_hair, brown_eyes, earrings, freckles, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body, is walking in a forest" \
  --height=912 \
  --width=704 \