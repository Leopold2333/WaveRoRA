project_path=$(pwd)
work_space="$project_path/main.py"

model="WaveRoRA"
seq_len=96
loss="mse"
random_seed=2025
e_layers=4
wavelet_layers=4

w_dim=64
d_model=512
d_ff=256
batch_size=32

# dataset
dataset_name="traffic"
root_path=data/traffic
file_name="${dataset_name}.csv"
dataset_type=custom
enc_in=862

gpu=0

is_training=$1

# for e_layers in 1 2 3
for wavelet_type in 'sym3'
do
    if [ $wavelet_type = "coif3" ]; then
        random_seed=2023
    elif [ $wavelet_type = "haar" ]; then
        random_seed=2024
    elif [ $wavelet_type = "sym3" ]; then
        random_seed=2025
    fi
    for pred_len in 96
    do
        if [ "$pred_len" -eq 96 ]; then
            lr="2e-3"
            dropout=0.3
        elif [ "$pred_len" -eq 192 ]; then
            lr="2e-3"
            dropout=0.3
        elif [ "$pred_len" -eq 336 ]; then
            lr="2e-3"
            dropout=0.2
        elif [ "$pred_len" -eq 720 ]; then
            lr="1e-3"
            dropout=0.3
        fi
        log_file="logs/Hyper/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}.log"
        python $work_space $model --is_training=$is_training --gpu=$gpu \
        --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
        --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm --embed_type=2 --rotary --gate \
        --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
        --e_layers=$e_layers --wavelet_layers=$wavelet_layers --wavelet_type=$wavelet_type --wavelet_mode=zero \
        --d_model=$d_model --d_ff=$d_ff --wavelet_dim=$w_dim --n_heads=8 --ks=3 --output_attention \
        --learning_rate=$lr --dropout=$dropout \
        # > $log_file 2>&1
    done
done