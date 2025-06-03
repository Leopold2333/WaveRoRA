project_path=$(pwd)
work_space="$project_path/main.py"

model="Autoformer"
seq_len=96
label_len=96
loss="mse"
random_seed=0
e_layers=2
d_layers=1


d_model=512
d_ff=2048
batch_size=32

# dataset
dataset_name="electricity"
root_path=data/electricity
file_name="${dataset_name}.csv"
dataset_type=custom
enc_in=321
dec_in=321
c_out=321

gpu=0

is_training=$1
at="RA"
for pred_len in 336 720 96 192
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
    python $work_space $model --is_training=$is_training --gpu=$gpu --patience=3 --train_epochs=10 \
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