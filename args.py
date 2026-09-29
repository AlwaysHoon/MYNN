import argparse

def main_args():
    parser = argparse.ArgumentParser(description="Basic Pytorch Neural Network Training")
    #parser.add_argument("--", type=, default=, help="")

    #Dataset
    parser.add_argument("--data",            type=str, required=True,           help="Dataset root directory")

    #model
    parser.add_argument("-p", "--pretrained", action="store_true",              help="Use Pretrained model")
    parser.add_argument("-a","--arch",       type=str, default="resnet18",      help="Model architecture")

    #Training
    parser.add_argument("--epochs",          type=int, default=100,             help="Number of training epochs")
    parser.add_argument("-b","--batch-size", type=int, default=128,             help="Mini-batch size")
    parser.add_argument("--lr",              type=float, default=0.01,          help="Initial learning rate")
    parser.add_argument("--momentum",        type=float, default=0.9,           help="SGD Momentum")
    parser.add_argument("--wd",              type=float, default=1e-4,          help="Weight Decay")

    # System
    parser.add_argument("--workers",         type=int, default=4,               help="Number of DataLoader workers")

    # Checkpoints
    parser.add_argument("--resume",          type=str, default="",              help="Checkpoint path")
    parser.add_argument("--save-path",       type=str, default="checkpoints/",  help="Directory for checkpoints")

    #etc
    parser.add_argument('--seed',            type=int, default=None,            help='seed for initializing training. ')

    return parser