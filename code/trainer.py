#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from lib.model import PriceGraph
from lib.model import output_layer
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PWD = os.path.dirname(os.path.realpath(__file__))
x_column = ['close', 'open', 'high', 'low', 'vol', 'amount']


def data_filter(dataset):
    sample_len = len(dataset['close_ems'])
    mask = [True] * sample_len
    for i in range(sample_len):
        for xi in x_column:
            s = dataset[xi+'_ems'][i]
            if s.shape != (20, 32):
                mask[i] = False
    return {k: np.array(list(v[mask])) for k, v in dataset.items()}


def load_season(season):
    with open(os.path.join('../dataset', '2019_S%s.pickle' % season), 'rb') as fp:
        return data_filter(pickle.load(fp))


def merge_season(dataset, season):
    data = load_season(season)
    for key in dataset.keys():
        dataset[key] = np.append(dataset[key], data[key], axis=0)
    return dataset


def load_year(year):
    with open(os.path.join('../dataset', '%s.pickle' % year), 'rb') as fp:
        return data_filter(pickle.load(fp))


def load_train(years):
    train = None
    for y in years:
        dataset = load_year(y)
        if train is None:
            train = dataset
            continue
        for key in dataset.keys():
            train[key] = np.append(train[key], dataset[key], axis=0)
    return train


def load_dataset():
    train = load_train(years)
    for s in range(1, Season):
        train = merge_season(train, s)
    test = load_season(Season)
    train['ts'] = np.array(train['close_t'])
    test['ts'] = np.array(test['close_t'])
    return train, test


class Trainer:
    def __init__(self, time_step, hidden_size, lr, batch_size=256, drop_ratio=0, split=20):
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.learning_rate = lr
        self.batch_size = batch_size
        self.drop_ratio = drop_ratio
        self.validation_ratio = split

        self.train, self.test = load_dataset()
        feature_size = self.train['close_ems'][0].shape[1]
        self.feature_size  = feature_size
        self.emtree = PriceGraph(feature_size, hidden_size, time_step, drop_ratio)
        self.output = output_layer(last_hidden_size=hidden_size, output_size=1)

        if torch.cuda.is_available():
            self.emtree = self.emtree.cuda()
            self.output = self.output.cuda()
        self.emtree_optim = optim.Adam(self.emtree.parameters(), lr, weight_decay=L2)
        self.output_optim = optim.Adam(self.output.parameters(), lr, weight_decay=L2)

        self.loss_func = nn.BCELoss()
        self.test_acc_max = 0
        self.model_name = '%s_%s_S%s_f%s_h%s_b%s_t%s' % (min(years), max(years), Season, self.feature_size, self.hidden_size, self.batch_size, self.time_step)
        self.epoch = 0

    def metrics(self, ori_y, results):
        accuracy = accuracy_score(ori_y, results)
        precision = precision_score(ori_y, results, labels=[1], average=None)[0]
        recall = recall_score(ori_y, results, labels=[1], average=None)[0]
        f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
        return accuracy, precision, recall, f1

    def train_minibatch(self, num_epochs):
        train_size = len(self.train['ts'])
        result = {}
        for epoch in range(num_epochs):
            loss_sum = 0
            y_predict = []
            # randomly select validation set
            validation_index = random.sample(range(train_size), int(train_size*self.validation_ratio/100.))
            validation_mask = np.array([False] * train_size)
            validation_mask[validation_index] = True
            train_data = {k: v[~validation_mask] for k, v in self.train.items()}
            val_data = {k: v[validation_mask] for k, v in self.train.items()}
            for i in range(0, len(train_data['ts']), self.batch_size):
                batch = [{yi: self.to_variable(train_data['%s_%s' % (xi, yi)][i:i+self.batch_size])
                          for yi in ['ems', 'ys', 'cis']} for xi in x_column]
                var_t = self.to_variable(train_data['ts'][i:i+self.batch_size])
                out1 = self.emtree(batch)
                out2 = self.output(out1)
                out3 = (out2 >= 0.5) + 0
                y_predict.extend(out3.data.cpu().numpy())

                loss = self.loss_func(out2, var_t)
                self.emtree_optim.zero_grad()
                self.output_optim.zero_grad()
                loss.backward()
                self.emtree_optim.step()
                self.output_optim.step()
                loss_sum += loss.data.item()

            print('\n--------------------------------------------------------------\n')
            print('epoch [%d] finished, the average loss is %f\n' % (epoch, loss_sum*self.batch_size/len(train_data['ts'])))

            print('Train:\n')
            accuracy, precision, recall, f1 = self.metrics(list(train_data['ts']), y_predict)
            print('Accuracy:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f\n' % (accuracy, precision, recall, f1))

            print('Validation:\n')
            validation_random = max([sum(val_data['ts'] == r) for r in [0, 1]]) * 1. / len(val_data['ts'])
            accuracy, precision, recall, f1 = self.validation(val_data)
            acc_max_diff = max(acc_max_diff, accuracy-validation_random)
            print('Accuracy:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f\n' % (accuracy, precision, recall, f1))
            print('Random:%.4f\tMaxAccDiff:%.6f\n' % (validation_random, acc_max_diff))

            print('Test:\n')
            test_random = max([sum(self.test['ts'] == r) for r in [0, 1]]) * 1. / len(self.test['ts'])
            accuracy, precision, recall, f1 = self.validation(self.test)
            acc_max = max(acc_max, accuracy)
            print('Accuracy:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f\n' % (accuracy, precision, recall, f1))
            print('Random:%.4f\tMaxAcc:%.4f\n' % (test_random, acc_max))

            if (epoch + 1) % interval == 0 or epoch + 1 == num_epochs:
                torch.save(self.emtree.state_dict(), '../models/emtree_%s_%s.model' % (self.model_name, str(epoch + 1)))
                torch.save(self.output.state_dict(), '../models/output_%s_%s.model' % (self.model_name, str(epoch + 1)))

    def validation(self, data):
        y_predict = []
        for i in range(0, len(data['ts']), self.batch_size):
            batch = [{yi: self.to_variable(data['%s_%s' % (xi, yi)][i:i+self.batch_size])
                      for yi in ['ems', 'ys', 'cis']} for xi in x_column]
            out1 = self.emtree(batch)
            out2 = self.output(out1)
            out3 = (out2 >= 0.5) + 0
            y_predict.extend(out3.data.cpu().numpy())
        return self.metrics(list(data['ts']), y_predict)

    def to_variable(self, x):
        if torch.cuda.is_available():
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def load_model(self, epoch):
        emtree_path = '../models/emtree_%s_%s.model' % (self.model_name, epoch)
        output_path = '../models/output_%s_%s.model' % (self.model_name, epoch)
        self.emtree.load_state_dict(torch.load(emtree_path, map_location=lambda storage, loc: storage))
        self.output.load_state_dict(torch.load(output_path, map_location=lambda storage, loc: storage))

    def load_model_2019(self, season):
        emtree_path = '../models/PriceGraph_emtree_S%s.model' % season
        output_path = '../models/PriceGraph_output_S%s.model' % season
        self.emtree.load_state_dict(torch.load(emtree_path, map_location=lambda storage, loc: storage))
        self.output.load_state_dict(torch.load(output_path, map_location=lambda storage, loc: storage))

    def test_data(self):
        data = self.test
        y_predict = []
        bs = self.batch_size
        for i in range(0, len(data['ts']), bs):
            batch = [{yi: self.to_variable(data['%s_%s' % (xi, yi)][i:i+bs])
                      for yi in ['ems', 'ys', 'cis']} for xi in x_column]
            out1 = self.emtree(batch)
            out2 = self.output(out1)
            out3 = (out2 >= 0.5) + 0
            y_predict.extend(out3.data.cpu().numpy())
        accuracy, precision, recall, f1 = self.metrics(list(data['ts']), y_predict)
        print('Accuracy:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f\n' % (accuracy, precision, recall, f1))


def getArgParser():
    parser = argparse.ArgumentParser(description='Train the price graph model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1000,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=256,
        help='the mini-batch size')
    parser.add_argument(
        '-ts', '--timestep', type=int, default=20,
        help='the length of time_step')
    parser.add_argument(
        '-hs', '--hiddensize', type=int, default=32,
        help='the length of hidden size')
    parser.add_argument(
        '-y', '--years', type=int, nargs='+',
        help='an integer for the accumulator')
    parser.add_argument(
        '-sn', '--season', type=int, default=1,
        help='the test season of 2019')
    parser.add_argument(
        '-dr', '--dropratio', type=int, default=0,
        help='the ratio of drop')
    parser.add_argument(
        '-s', '--split', type=int, default=30,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=50,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-l2', '--l2rate', type=float, default=0,
        help='L2 penalty lambda')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '--test_2019', action='store_true',
        help='test with models for 2019')
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    time_step = args.timestep
    hidden_size = args.hiddensize
    drop_ratio = args.dropratio
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    L2 = args.l2rate
    Season = args.season
    years = args.years

    print(args)
    trainer = Trainer(time_step, hidden_size, lr, batch_size, drop_ratio, split)
    if args.test:
        if args.test_2019:
            trainer.load_model_2019(Season)
        else:
            trainer.load_model(num_epochs)
        trainer.test_data()
    else:
        trainer.train_minibatch(num_epochs)
