def add_decompose_parser(parser):
        parser.add_argument('--decompose',                  default=False,  action='store_true',
                                                                            help='decomposition')
        parser.add_argument('--kernel_size',        type=int,   default=25,             help='decomposition-kernel of AVGPool')