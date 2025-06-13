project_path=$(pwd)
work_space="$project_path/main.py"

model="WaveRoRA"
seq_len=96
loss="mse"
random_seed=2025
e_layers=2
wavelet_layers=3

w_dim=64
d_ff=256
batch_size=32

# dataset
root_path=data/ETT-small
enc_in=7
gpu=0
is_training=$1

wavelet_type=sym3
d_model=256
dataset_name=ETTh1
file_name=ETTh1.csv
dataset_type=ETTh1
for pred_len in 96 192 336 720
do
    if [ "$pred_len" -eq 96 ]; then
        lr="1e-4"
        dropout=0.3
        e_layers=1
        wavelet_layers=3
    elif [ "$pred_len" -eq 192 ]; then
        lr="4e-5"
        dropout=0.0
        e_layers=1
        wavelet_layers=3
    elif [ "$pred_len" -eq 336 ]; then
        lr="4e-5"
        dropout=0.0
        e_layers=1
        wavelet_layers=3
    elif [ "$pred_len" -eq 720 ]; then
        lr="4e-5"
        dropout=0.0
        e_layers=2
        wavelet_layers=4
    fi
    log_file="logs/LTSF/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}_wl${wavelet_layers}_${wavelet_type}.log"
    python $work_space $model --is_training=$is_training --gpu=$gpu --patience=3 \
    --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
    --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm --embed_type=0 --rotary --gate --residual \
    --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
    --e_layers=$e_layers --wavelet_layers=$wavelet_layers --wavelet_type=$wavelet_type --wavelet_mode=zero \
    --d_model=$d_model --d_ff=$d_ff --wavelet_dim=$w_dim \
    --learning_rate=$lr --dropout=$dropout \
    > $log_file 2>&1
done

wavelet_type=coif3
d_model=256
dataset_name=ETTh2
file_name=ETTh2.csv
dataset_type=ETTh2
for pred_len in 96 192 336 720
do
    if [ "$pred_len" -eq 96 ]; then
        lr="4e-5"
        dropout=0.1
        e_layers=2
        wavelet_layers=5
    elif [ "$pred_len" -eq 192 ]; then
        lr="4e-5"
        dropout=0.0
        e_layers=2
        wavelet_layers=5
    elif [ "$pred_len" -eq 336 ]; then
        lr="4e-5"
        dropout=0.0
        e_layers=2
        wavelet_layers=5
    elif [ "$pred_len" -eq 720 ]; then
        lr="1e-4"
        dropout=0.1
        e_layers=2
        wavelet_layers=3
    fi
    log_file="logs/LTSF/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}_wl${wavelet_layers}_${wavelet_type}.log"
    python $work_space $model --is_training=$is_training --gpu=$gpu --patience=3 \
    --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
    --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm --embed_type=0 --rotary --gate --residual \
    --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
    --e_layers=$e_layers --wavelet_layers=$wavelet_layers --wavelet_type=$wavelet_type --wavelet_mode=zero \
    --d_model=$d_model --d_ff=$d_ff --wavelet_dim=$w_dim --ks=3 \
    --learning_rate=$lr --dropout=$dropout \
    > $log_file 2>&1
done