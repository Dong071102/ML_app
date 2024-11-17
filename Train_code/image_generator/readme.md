```bash
DATASET_NAME="dongOi071102/meme-image-no-text"
```

```bash
accelerate launch train_text_to_image_lora_prior.py \
 --mixed_precision="fp16" \
 --dataset_name=$DATASET_NAME --caption_column="text" \
 --resolution=512 \
 --train_batch_size=8 \
 --num_train_epochs=100 --checkpointing_steps=500 \
 --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
 --seed=42 \
 --rank=8 \
 --validation_prompt="a cartoon character sitting at a desk with a computer" \
 --report_to="wandb" \
 --push_to_hub \
 --output_dir="wuerstchen-prior-meme-image-no-text-lora-v1"\
 --weight_dtype="float32"
```
