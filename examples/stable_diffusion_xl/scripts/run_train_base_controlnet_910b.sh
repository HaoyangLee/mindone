# export MS_PYNATIVE_GE=1

save_path=runs/fill1k
mkdir -p $save_path

python train.py \
    --ms_mode 0 \
    --config configs/training/sd_xl_base_finetune_controlnet_910b.yaml \
    --data_path data/fill1k \
    --weight checkpoints/sd_xl_base_1.0_ms_controlnet_init.ckpt \
    --save_ckpt_interval 10000 \
    --per_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_step 200000 \
    --max_num_ckpt 5 \
    > $save_path/train.log 2>&1 &
