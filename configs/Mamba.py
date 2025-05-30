def add_mamba_parser(parser):
    if parser.get_default('d_model') is None:
        # token embedding
        parser.add_argument('--d_model',        type=int,       default=128,         help='Sequence Elements embedding dimension')
        parser.add_argument('--d_ff',           type=int,       default=256,        help='Second Embedded representation')
    
    # mamba block
    parser.add_argument('--bi_dir',         type=int,       default=0,          help='use bidirectional Mamba?')
    parser.add_argument('--d_state',        type=int,       default=16,         help='d_state parameter of Mamba')
    parser.add_argument('--d_conv',         type=int,       default=2,          help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact',         type=int,       default=1,          help='expand factor parameter of Mamba')

    if parser.get_default('e_layers') is None:
        parser.add_argument('--e_layers',       type=int,       default=1,          help='layers of encoder')
        parser.add_argument('--dropout',        type=float,     default=0.2,        help='dropout')
        parser.add_argument('--activation',     type=str,       default='gelu',     help='activation')
        parser.add_argument('--residual',       type=int,       default=1,          help='residual connection?')
