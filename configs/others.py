def add_others_parser(parser):
    # wavelet
    parser.add_argument("--wavelet_layers", type=int, default=5, help="the number of wavelet layer")
    parser.add_argument("--wavelet_type", type=str, default='haar', help='the wavelet function')