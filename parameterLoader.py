import argparse
from argparse import Namespace

def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption_rate', type=float, default=0.2)

    # Network Settings
    # Overall
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_vocab', type=int, default=11, help='5 | 11')
    parser.add_argument('--PAD', type=int, default=0)
    parser.add_argument('--MASK', type=int, default=1)
    # Graph Encoder
    parser.add_argument('--graph_n_layers', type=int, default=3) # 12
    parser.add_argument('--graph_d_model', type=int, default=64) # 64
    parser.add_argument('--graph_n_head', type=int, default=8)
    parser.add_argument('--graph_d_ff', type=int, default=64) # 128
    parser.add_argument('--maxL', type=int, default=15) # nas101:7 nas301: 15
    parser.add_argument('--graph_dropout', type=float, default=0.1)
    parser.add_argument('--weight_scale', type=float, default=0.1, help='0.01 | 0.1')

    # Cross_Attention
    parser.add_argument('--pair_n_layers', type=int, default=24) # 24
    parser.add_argument('--pair_d_model', type=int, default=64) # 64
    parser.add_argument('--pair_n_head', type=int, default=8)
    parser.add_argument('--pair_d_ff', type=int, default=64) # 128
    parser.add_argument('--pair_dropout', type=float, default=0.1)

    # CLS
    parser.add_argument('--cls_dropout', type=float, default=0.1)
    # Tied weight
    parser.add_argument('--tied_weights', action='store_true')

    # Device
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=4)

    # Data Loading
    parser.add_argument('--dataset', type=str, default="data/darts/")
    parser.add_argument('--train_data', type=str, default="data/darts/train.pt")
    parser.add_argument('--train_pair', type=str, default="data/darts/train_pairs.pt")
    parser.add_argument('--valid_data', type=str, default="data/darts/test.pt")
    parser.add_argument('--valid_pair', type=str, default="data/darts/test_pairs.pt")
    parser.add_argument('--mini', action="store_true")

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=100)

    # Loss Parameters
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Training Parameters
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument('--batch_size', type=int, default=64) # 1024
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--check_point_min', type=int, default=0)
    parser.add_argument('--check_point_freq', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='pace/') # 'pace_nodagseq/' 'pace_nomask/'
    parser.add_argument('--not_save_each_epoch', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default='model/')

    # Operations
    parser.add_argument('--do_train', action='store_true')

    args = parser.parse_args()
    args.save_each_epoch = not args.not_save_each_epoch
    
    args.gnn = Namespace()
    #args.gnn.indim = args.graph_d_model
    #args.gnn.outdim = args.graph_d_model
    args.gnn.indim = 64 # 64 (32)
    args.gnn.outdim = 64 # 64 (32)
    args.gnn.scale = args.weight_scale

    args.graph_encoder = Namespace()
    args.graph_encoder.n_layers = args.graph_n_layers
    args.graph_encoder.d_model = args.graph_d_model
    args.graph_encoder.n_head = args.graph_n_head
    args.graph_encoder.d_ff = args.graph_d_ff
    args.graph_encoder.dropout = args.graph_dropout
    args.graph_encoder.n_vocab = args.n_vocab
    args.graph_encoder.maxL = args.maxL
    
    args.cross_attention = Namespace()
    args.cross_attention.n_layers = args.pair_n_layers
    args.cross_attention.d_model = args.pair_d_model
    args.cross_attention.n_head = args.pair_n_head
    args.cross_attention.d_ff = args.pair_d_ff
    args.cross_attention.n_token_type = 2
    args.cross_attention.dropout = args.pair_dropout

    args.cls = Namespace()
    args.cls.d_model = args.pair_d_model
    args.cls.n_vocab = args.n_vocab
    args.cls.dropout = args.cls_dropout

    print(args)
    return args