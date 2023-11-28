export MS_PYNATIVE_GE=1

python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --guidance_scale 9.0 \
  --device_target Ascend \
  --controlnet_mode canny \
  --control_image_path /PATH TO/dog2.png \
  --prompt "cute dog, best quality, extremely detailed"   \
