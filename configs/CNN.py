def add_cnn_parser(parser):
    # CNN
    parser.add_argument("--residual_channels",  type=int,   default=32,     help='res connection')
    parser.add_argument("--skip_channels",      type=int,   default=64,     help="skip channels")
    parser.add_argument("--end_channels",       type=int,   default=128,     help="skip channels")
    parser.add_argument("--dilation_exponential",  type=int,   default=2,     help='dilation exponential factor')