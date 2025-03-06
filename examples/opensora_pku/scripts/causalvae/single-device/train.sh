python opensora/train/train_causalvae.py \
    --exp_name "25x256x256" \
    --model_name WFVAE \
    --model_config scripts/causalvae/wfvae_8dim.json \
    --train_batch_size 1 \
    --precision fp32 \
    --max_steps 100000 \
    --save_steps 2000 \
    --output_dir results/causalvae \
    --video_path datasets/UCF-101 \
    --data_file_path datasets/ucf101_train.csv \
    --video_num_frames 25 \
    --resolution 256 \
    --dataloader_num_workers 8 \
    --start_learning_rate 1e-5 \
    --lr_scheduler constant \
    --optim adamw \
    --betas 0.9 0.999 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --init_loss_scale 65536 \
    --jit_level "O0" \
    --use_discriminator True \
    --use_ema False \
    --ema_decay 0.999 \
    --perceptual_weight 0.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --disc_start 0 \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --print_losses
