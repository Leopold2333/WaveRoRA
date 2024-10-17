project_path=$(pwd)
work_space="$project_path/main.py"

model="iTransformer"
seq_len=96
loss="mse"
random_seed=2025
e_layers=3

d_model=128
d_ff=256
batch_size=32

# dataset
dataset_name="electricity"
root_path=data/electricity
file_name="${dataset_name}.csv"
dataset_type=custom
enc_in=321

gpu=0

lr="2e-4"
dropout=0.1

is_training=$1

for pred_len in 96
do
    log_file="logs/${model}/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})_el${e_layers}.log"
    python $work_space $model --is_training=$is_training --gpu=$gpu \
    --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
    --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in --use_norm \
    --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
    --e_layers=$e_layers \
    --d_model=$d_model --d_ff=$d_ff \
    --learning_rate=$lr --dropout=$dropout \
    > $log_file 2>&1
done