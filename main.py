import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import random
from tqdm import tqdm
import time
import argparse
import os
import io
import sys
import logging


from exp import exp_review_embedding, exp_rating


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/ciao/preprocess/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--dataset', default='dataset.pkl', help='pkl file dataset')
parser.add_argument('--datalist', default='ranklist.pkl', help='pkl file list')
parser.add_argument('--origindf', default='original_df.pkl', help='original dataframe')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=384, help='the dimension of embedding')
parser.add_argument('--epochs', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', default=5, help='early stop patience')
parser.add_argument('--save_path', default='./outputs', help='save path')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('--exp', default='rating', help='rating | review')
parser.add_argument('--metrics', default='mae', help='mae | hit')
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.FileHandler(os.path.join(args.save_path, "log.txt"))
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

file_error_handler = logging.FileHandler(os.path.join(args.save_path, "error.txt"))
file_error_handler.setLevel(logging.ERROR)
logger.addHandler(file_error_handler)

logger.info(args)


# seed
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def main():

    if args.exp == 'review':
        Exp = exp_review_embedding.GraphRec
    elif args.exp == 'rating':
        Exp = exp_rating.GraphRec

    exp = Exp(args)

    if args.metrics == 'hit':
        exp.train_and_infer_by_hit()
    elif args.metrics == 'mae':
        exp.train_and_infer_by_mse()


if __name__ == "__main__":
    main()