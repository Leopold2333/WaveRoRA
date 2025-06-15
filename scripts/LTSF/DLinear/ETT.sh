project_path=$(pwd)
work_space="$project_path/main.py"

model="DLinear"
seq_len=96
loss="mse"
random_seed=2024

batch_size=32

# dataset
dataset_name="ETTh1"
root_path=data/ETT-small
file_name="${dataset_name}.csv"
dataset_type=ETTh1
enc_in=7

gpu=0

lr="1e-4"
dropout=0.2

is_training=$1

for dataset_name in ETTh1 ETTh2
do
    file_name="${dataset_name}.csv"
    for pred_len in 96 192 336 720
    do
        log_file="logs/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len}).log"
        python $work_space $model --is_training=$is_training --gpu=$gpu \
        --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
        --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in \
        --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
        --e_layers=$e_layers \
        --learning_rate=$lr --dropout=$dropout \
        > $log_file 2>&1
    done
done

dataset_type=ETTm1
for dataset_name in ETTm1 ETTm2
do
    file_name="${dataset_name}.csv"
    for pred_len in 96 192 336 720
    do
        log_file="logs/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len}).log"
        python $work_space $model --is_training=$is_training --gpu=$gpu \
        --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
        --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in \
        --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
        --e_layers=$e_layers \
        --learning_rate=$lr --dropout=$dropout \
        > $log_file 2>&1
    done
done