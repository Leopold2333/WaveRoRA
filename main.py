import argparse
import torch
import random
import numpy as np
import optuna
import sys
from utils.model_params import model_parser_dict, model_basic_parsers
from data_provider.dataloader import load_data
from trainer.long_term_forecasting import LTSF_Trainer

def objective(trial, args, search_params, data):
    """
    Single parameter selection.

    :param trial: An optuna parameter selection trial.
    :param args: Hyperparameters from 'argparse'.

    :return: The MSE/MAE result for each Optuna trial to record the best result.
    """
    # fixed random seed
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parameters settings
    # args.learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.0016, step=0.00007)
    # args.dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.05)
    # args.n_heads = trial.suggest_int("n_heads", 2, 16, step=2)
    # args.wavelet_layers = trial.suggest_int("wavelet_layers", 2, 5, step=1)
    
    task = '{}({})_{}_{}_loss({})'.format(
        args.dataset_name,          # dataset
        args.task,
        args.seq_len,               # look-back window size
        args.pred_len,              # prediction horizons
        args.loss
    )

    setting = '{0}_bs{1}{2}{3}{4}{5}{6}{7}{8}'.format(
        args.model,                 # 0. model name
        args.batch_size,            # 1. batch-size
        f'_el{args.e_layers}' if hasattr(args, 'e_layers') else '',             # 2. layers of encoder
        f'_(dm{args.d_model}+df{args.d_ff})' if hasattr(args, 'd_model') else '',           # 3. the latent state dimension
        f'_nh{args.n_heads}' if hasattr(args, 'n_heads') else '',               # 4. multi-heads
        f'_(dc{args.d_conv}+ds{args.d_state})' if hasattr(args, 'd_conv') else '',          # 5. mamba block
        f'_dp{args.dropout:.2f}' if hasattr(args, 'dropout') else '',           # 6. dropout
        f'_(pl{args.patch_len}+st{args.stride})' if hasattr(args, 'patch_len') else '',     # 7. patching
        f'_({args.wavelet_layers}-{args.wavelet_dim}-{args.wavelet_type})' if hasattr(args, 'wavelet_type') else '',   # 8. wavelet
    )

    print(f'>>>>>> Searching Params <<<<<<')
    for param_name in search_params:
        param_value = getattr(args, param_name)
        print(f'{param_name}={param_value}')

    engine = LTSF_Trainer(args, task, setting)

    if args.is_training == 1:
        engine.train(data=data)
    if args.is_training >=0:
        mse = engine.test(test_loader=data['test_loader'], test_dataset=data['test'])
        return mse
    else:
        engine.predict(pred_loader=data['pred_loader'])
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer & non-Transformer family for Time Series Forecasting & Anomaly Detection')
    assert len(sys.argv) > 1, "Expecting arg index <1> as the model's name, with other places as normal arguments!"

    print(sys.argv[1])
    model_basic_parsers(parser=parser)
    for p in model_parser_dict[sys.argv[1]]:
        p(parser=parser)
    
    args, model_name = parser.parse_known_args()
    args.model = model_name[0]
    
    # find usable GPUs, otherwise cpu
    if torch.cuda.is_available() and not args.use_cpu:
        args.device='cuda:{}'.format(args.gpu)
    else:
        args.device='cpu'

    # Grad Searching
    search_space = {
        "weather": {
            "learning_rate": [1e-4, 4e-5], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
        "traffic": {
            "learning_rate": [1e-3, 1.5e-3, 8e-4], 
            "dropout": [0.2, 0.3, 0.1],
        },
        "electricity": {
            "learning_rate": [1e-3], 
            "dropout": [0.2,  0.1]
        },
        "ETTh1": {
            "learning_rate": [1e-4, 4e-5], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
        "ETTh2": {
            "learning_rate": [1e-4, 4e-5], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
        "ETTm1": {
            "learning_rate": [1e-4, 4e-5], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
        "ETTm2": {
            "learning_rate": [1e-4, 4e-5], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
        "solar": {
            "learning_rate": [5e-4, 2e-4, 1e-3], 
            "dropout": [0.1, 0.2, 0.3, 0.0]
        },
    }
    params = list(search_space[args.dataset_name].keys())
    print('>>>>>> Args in experiment: <<<<<<')
    for k, v in sorted(vars(args).items()):
        if k not in params:
            print(f'{k}={v}')
    data = load_data(args)
    # study = optuna.create_study(direction='minimize')
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.GridSampler(search_space[args.dataset_name]))
    # bind 'args' to 'objective' function
    objective_with_args = lambda trial: objective(trial, args, params, data)
    study.optimize(objective_with_args, n_trials=1)
    # print the best parameters and metrics
    print('Best parameters:', study.best_params)
    print('Best score:', study.best_value)
