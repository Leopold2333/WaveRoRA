def add_patch_parser(parser):
    parser.add_argument('--patch_len',          type=int,   default=16,             help='patch length')
    parser.add_argument('--stride',             type=int,   default=8,              help='stride, half or full length of patch_len')
    parser.add_argument('--multi_patch',                    default=False,  action="store_true", help='patch length')
    parser.add_argument('--multi_scale_list',   type=list,  default=[8, 4, 2],       help='patch length list')
    parser.add_argument('--padding_patch',      type=str,   default='end',          help='None: None; end: padding on the end')

    # # Deform
    # parser.add_argument('--deform_patch',                   default=False,  action="store_true", help='deform_patch')
    # parser.add_argument('--deform_range',       type=float, default=0.5, help='deform_range')
    # parser.add_argument('--lambda_',            type=float, default=1e-1, help='PaEn Weight')
    # parser.add_argument('--r',                  type=float, default=1e-2, help='Parameter of PaEn')
    # # Swin
    # parser.add_argument('--mlp_ratio',          type=float, default=1.0, help='mlp_ratio')
    # parser.add_argument('--window_size',        type=int,   default=6, help='window_size')
    # parser.add_argument('--shift_size',         type=int,   default=3, help='shift_size')
    # parser.add_argument('--weight_decay',       type=float, default=1e-3, help='window_size')
