def add_linear_parser(parser):
    parser.add_argument('--individual',                     default=False, action='store_true', 
                        help='DLinear: a linear layer for each variate(channel) individually?')