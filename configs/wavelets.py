def add_wavelet_parser(parser):
    # wavelet
    parser.add_argument("--wavelet_layers", type=int,   default=5,      help="the number of wavelet layer")
    parser.add_argument("--wavelet_type",   type=str,   default='haar', help='the wavelet function')
    parser.add_argument("--wavelet_mode",   type=str,   default='zero', help='the wavelet mode')
    parser.add_argument("--wavelet_dim",    type=int,   default=64,     help='dimension of each wavelet series')
    parser.add_argument("--domain",         type=str,   default='W',    help="W: wavelet; T: time; F: frequency")
