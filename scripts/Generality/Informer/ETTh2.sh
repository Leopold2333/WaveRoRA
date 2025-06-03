project_path=$(pwd)
work_space="$project_path/main.py"

model="Informer"
seq_len=96
label_len=48
loss="mse"
random_seed=0
e_layers=3
d_layers=2


d_model=256
d_ff=1024
batch_size=32

# dataset
dataset_name="ETTh2"
root_path=data/ETT-small
file_name="${dataset_name}.csv"
dataset_type=ETTh2
enc_in=7
dec_in=7
c_out=7

gpu=0

is_training=$1
at="RA"
for pred_len in 96 192 336 720
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
    log_file="logs/Generality/${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})${at}.log"
    python $work_space $model --is_training=$is_training --gpu=$gpu --patience=3 \
    --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
    --seq_len=$seq_len --label_len=$label_len --pred_len=$pred_len --enc_in=$enc_in --dec_in=$dec_in --c_out=$c_out \
    --embed_type=3 --attn_type=$at \
    --dataset_name=$dataset_name --file_name=$file_name --root_path=$root_path --dataset_type=$dataset_type \
    --e_layers=$e_layers --d_layers=$d_layers \
    --d_model=$d_model --d_ff=$d_ff --n_heads=8 --factor=5 --distil --router_num=10 \
    --learning_rate=$lr --dropout=$dropout \
    > $log_file 2>&1
done
# done