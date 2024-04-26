import os
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

from model import GraphRecRating

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.FileHandler("./outputs/log.txt")
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

file_error_handler = logging.FileHandler("./outputs/error.txt")
file_error_handler.setLevel(logging.ERROR)
logger.addHandler(file_error_handler)


class GraphRec:
    def __init__(self, args):
        self.args = args

    def _load_data(self):
        data_path = f'{self.args.data_path}{self.args.dataset}'
        list_path = f'{self.args.data_path}{self.args.datalist}'
        data_file = open(data_path, 'rb')
        list_file = open(list_path, 'rb')

        history_u, history_i, history_ur, history_ir, history_ue, history_ie, \
        train_u, train_i, train_r, valid_u, valid_i, valid_r, \
        test_u, test_i, test_r, social_neighbor, ratings =  pickle.load(data_file)
        valid_rank_data, test_rank_data = pickle.load(list_file)

        return history_u, history_i, history_ur, history_ir, history_ue, history_ie, \
            train_u, train_i, train_r, valid_u, valid_i, valid_r, \
            test_u, test_i, test_r, social_neighbor, ratings, \
            valid_rank_data, test_rank_data
    

    def train_and_infer_by_hit(self):
        # data
        history_u, history_i, history_ur, history_ir, history_ue, history_ie, \
        train_u, train_i, train_r, valid_u, valid_i, valid_r, \
        test_u, test_i, test_r, social_neighbor, ratings, \
        valid_rank_data, test_rank_data = self._load_data()

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i),
                                              torch.FloatTensor(train_r))
        validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_i),
                                                torch.FloatTensor(valid_r))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_i),
                                                torch.FloatTensor(test_r))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(validset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
        
        num_users = history_u.__len__()
        num_items = history_i.__len__()
        num_ratings = ratings.__len__()
        
        # model
        model = GraphRecRating.Model(num_users, num_items, num_ratings, history_u, history_i, history_ur,\
                                     history_ir, self.args.embed_dim, social_neighbor, cuda=self.args.device).to(self.args.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr, alpha=0.9)
        scheduler = StepLR(optimizer, step_size = self.args.lr_dc_step, gamma = self.args.lr_dc)

        # train
        best_hits = 0.0
        endure_count = 0
        best_test_hits = 0

        running_loss = 0.0
        model.train()
        for epoch in range(1, self.args.epochs+1):

            for i, data in enumerate(train_loader):
                batch_nodes_u, batch_nodes_i, batch_ratings = data
                optimizer.zero_grad()
                loss = model.loss(batch_nodes_u.to(self.args.device), batch_nodes_i.to(self.args.device), batch_ratings.to(self.args.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    logger.info(f'[Epoch: {epoch}, Iter: {i}] Train Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
        
                val_hits = self.rank_test(model, valid_rank_data)
            
            scheduler.step()

            # early stopping
            if best_hits < val_hits:
                best_hits = val_hits
                endure_count = 0
                torch.save(model, os.path.join(self.args.save_path, 'best_checkpoint.pt'))
            else:
                endure_count += 1
            logger.info(f"val HITS@10:{val_hits:.4f}")

            test_hits = self.rank_test(model, test_rank_data)
            if test_hits > best_test_hits:
                best_test_hits = test_hits
            logger.info(f"best test HITS@10:{best_test_hits:.4f}")

            if endure_count > self.args.patience:
                logger.info("early stopping...")
                break
        
        test_hits = self.rank_test(model, test_rank_data)
        logger.info(f"test HITS@10:{test_hits:.4f}")
        logger.info(f"best test HITS@10:{best_test_hits:.4f}")


    def train_and_infer_by_mse(self):
        # data
        history_u, history_i, history_ur, history_ir, history_ue, history_ie, \
        train_u, train_i, train_r, valid_u, valid_i, valid_r, \
        test_u, test_i, test_r, social_neighbor, ratings, \
        valid_rank_data, test_rank_data = self._load_data()

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i),
                                              torch.FloatTensor(train_r))
        validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_i),
                                                torch.FloatTensor(valid_r))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_i),
                                                torch.FloatTensor(test_r))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(validset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
        
        num_users = history_u.__len__()
        num_items = history_i.__len__()
        num_ratings = ratings.__len__()
        
        # model
        model = GraphRecRating.Model(num_users, num_items, num_ratings, history_u, history_i, history_ur,\
                                     history_ir, self.args.embed_dim, social_neighbor, cuda=self.args.device).to(self.args.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr, alpha=0.9)
        scheduler = StepLR(optimizer, step_size = self.args.lr_dc_step, gamma = self.args.lr_dc)

        # train
        best_mae = 9999.0
        best_test_mae = 9999.0
        endure_count = 0

        running_loss = 0.0
        model.train()
        for epoch in range(1, self.args.epochs+1):

            for i, data in enumerate(train_loader):
                batch_nodes_u, batch_nodes_i, batch_ratings = data
                optimizer.zero_grad()
                loss = model.loss(batch_nodes_u.to(self.args.device), batch_nodes_i.to(self.args.device), batch_ratings.to(self.args.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    logger.info(f'[Epoch: {epoch}, Iter: {i}] Train Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
        
                val_rmse, val_mae = self.test(model, val_loader)
            
            scheduler.step()

            # early stopping
            if best_mae > val_mae:
                best_mae = val_mae
                endure_count = 0
                torch.save(model, os.path.join(self.args.save_path, 'best_checkpoint.pt'))
            else:
                endure_count += 1
            logger.info(f"val rmse: {val_rmse:.4f}, val mae: {val_mae:.4f}")

            test_rmse, test_mae = self.test(model, test_loader)
            if best_test_mae > test_mae:
                best_test_mae = test_mae
            logger.info(f"best test mae:{best_test_mae:.4f}")

            if endure_count > self.args.patience:
                logger.info("early stopping...")
                break
        
        test_rmse, test_mae = self.test(model, test_loader)
        logger.info(f"test rmse: {test_rmse:.4f}, test mae: {test_mae:.4f}")

    
    def rank_test(self, model, test_data):
        model.eval()
        rank_list = []
        for u in test_data:
            item = test_data[u]
            neg_list = item['neg'].tolist()
            user = torch.LongTensor([u]*len(neg_list)).to(self.args.device)
            pos = torch.LongTensor([item['pos']]).to(self.args.device)
            neg = torch.LongTensor(neg_list).to(self.args.device)
            with torch.no_grad():
                scores_neg = model(user, neg)
                scores_pos = model(user[:1], pos)
                rank = np.argsort(-np.hstack((scores_pos.cpu().numpy(), scores_neg.cpu().numpy())))
                rank = rank[:10]
                rank_list.append(int(0 in rank))

        return np.mean(rank_list)
    

    def test(self, model, test_loader):
        model.eval()
        pred = []
        target = []
        with torch.no_grad():
            for test_u, test_i, test_ratings in test_loader:
                test_u, test_i, test_ratings = test_u.to(self.args.device), test_i.to(self.args.device), test_ratings.to(self.args.device)
                scores = model(test_u, test_i)
                pred.append(list(scores.cpu().numpy()))
                target.append(list(test_ratings.cpu().numpy()))
        pred = np.array(sum(pred, []))
        target = np.array(sum(target, []))
        rmse = sqrt(mean_squared_error(pred, target))
        mae = mean_absolute_error(pred, target)

        return rmse, mae