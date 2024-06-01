Time=$(date "+%Y%m%d-%H%M%S")
cd /userhome/Research_HUB/KIP-frame
config_name=$1
python KIP_frame.py --config 'config_dir/'${config_name}'.ini' >log_dir/Few-shot/${config_name}_fewshot_${Time}.log 2>&1
