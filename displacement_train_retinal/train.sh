echo "start training"
# 获取当前脚本文件的路径
script_dir=$(dirname "$0")
python train.py > "$script_dir/save/experiments/benchmark7.txt"