export GLOG_v=2
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1: detail, 0: simple
export DEVICE_ID=7

output_path='outputs/train'

if [ ! -d "$output_path" ]; then
mkdir -p $output_path
echo "Created directory to save output: $output_path"
fi

nohup python -u train.py \
    > $output_path/train.log 2>&1 &
