project_path=$(pwd)
work_space="$project_path/main.py"

model="WaveFormer"
seq_len=96
loss="mse"
random_seed=2024
e_layers=2
wavelet_layers=5

w_dim=64
d_model=256
d_ff=256
batch_size=32

# dataset
root_path=data/ETT-small
enc_in=7

gpu=0

lr="1e-4"
dropout=0.2

is_training=$1

for wavelet_type in "coif3" "haar"
do
    if [ $wavelet_type = "coif3" ]; then
        random_seed=2023
    elif [ $wavelet_type = "haar" ]; then
        random_seed=2024
    elif [ $wavelet_type = "sym3" ]; then
        random_seed=2025
    fi
    for dataset_name in ETTh1 ETTm1
    do
        file_name="${dataset_name}.csv"
        dataset_type=$dataset_name
        for pred_len in 96 192 336 720
        do
            log_file="logs/W/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}.log"
            python $work_space $model --is_training=$is_training --gpu=$gpu \
            --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
            --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm --embed_type=0 --rotary --gate --residual \
            --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
            --e_layers=$e_layers --wavelet_layers=$wavelet_layers --wavelet_type=$wavelet_type --wavelet_mode=zero \
            --d_model=$d_model --d_ff=$d_ff --wavelet_dim=$w_dim \
            --learning_rate=$lr --dropout=$dropout \
            > $log_file 2>&1
        done
    done
done

wavelet_layers=4
for wavelet_type in "coif3" "haar"
do
    if [ $wavelet_type = "coif3" ]; then
        random_seed=2023
    elif [ $wavelet_type = "haar" ]; then
        random_seed=2024
    elif [ $wavelet_type = "sym3" ]; then
        random_seed=2025
    fi
    for dataset_name in ETTh2 ETTm2
    do
        file_name="${dataset_name}.csv"
        dataset_type=$dataset_name
        for pred_len in 96 192 336 720
        do
            log_file="logs/M/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}.log"
            python $work_space $model --is_training=$is_training --gpu=$gpu \
            --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
            --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm --embed_type=0 --rotary --gate --residual \
            --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
            --e_layers=$e_layers --wavelet_layers=$wavelet_layers --wavelet_type=$wavelet_type --wavelet_mode=zero \
            --d_model=$d_model --d_ff=$d_ff --wavelet_dim=$w_dim \
            --learning_rate=$lr --dropout=$dropout \
            > $log_file 2>&1
        done
    done
done