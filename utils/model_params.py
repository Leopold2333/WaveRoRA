from configs import add_task_parser, add_dataset_parser, add_optim_parser, add_gpu_parser
from configs import add_linear_parser, add_transformer_parser, add_patch_parser, add_mamba_parser, \
                    add_wavelet_parser, add_cnn_parser, add_decompose_parser

model_parser_dict={
    'WaveRoRA': [add_transformer_parser, add_wavelet_parser],
    'MAWNO': [add_transformer_parser],
    'iTransformer': [add_transformer_parser],
    'PatchTST': [add_transformer_parser, add_patch_parser, add_decompose_parser],
    'DLinear': [add_linear_parser],
    'MambaFormer': [add_transformer_parser, add_mamba_parser]
}

def model_basic_parsers(parser):
    add_task_parser(parser=parser)
    add_dataset_parser(parser=parser)
    add_optim_parser(parser=parser)
    add_gpu_parser(parser=parser)