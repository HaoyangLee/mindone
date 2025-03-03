export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=8000 --log_dir="./lora_finetune_logs" \
  scripts/train.py \
   --config configs/finetune/mixkit_256x256x29.yaml \
   --train.save.use-lora True \
   --train.save.lora-rank 64 \
