import logging
import argparse
from urllib import parse
from model import RelModel
from config import Config
from utils import logger
from train import train

parser = argparse.ArgumentParser()

# common args
parser.add_argument('--index', type=str, default="debug")
parser.add_argument('--corpus_type', type=str, default="WebNLG", help="NYT, WebNLG, NYT-star, WebNLG-star")

# training args
parser.add_argument('--device-id', type=int, default=0)
parser.add_argument('--epoch-num', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=6)

# model enchanced args
parser.add_argument('--use-symmetries', type=str, default=None)
parser.add_argument('--use-feature-enchanced', action='store_true')
parser.add_argument('--use-negative-mask', action='store_true')

# component args
parser.add_argument('--set-rel-level', type=str, default='maxpooling',  help="cls_map, avg_map, maxpooling")
parser.add_argument('--set-ent-level', type=str, default='sent', help="sent, sent_enchanced, rel_maxpooling")
parser.add_argument('--set-table-calc', type=str, default='mul', help="mul, biaffine")

# loss weight
parser.add_argument('--tab-loss', type=float, default=10.0)
parser.add_argument('--rel-loss', type=float, default=0.0)
parser.add_argument('--ent-loss', type=float, default=0.0)
parser.add_argument('--cor-loss', type=float, default=0.0)

if __name__ == "__main__":

    args = parser.parse_args()
    config = Config(args)

    # config logger
    logger.set_logger(save=True, log_path=config.log_path)
    logging.info(config.get_args_info())
    logging.info(config.get_config_info())

    # model
    model = RelModel(config)

    # train
    train(model, config)

    # other
    print("done")
