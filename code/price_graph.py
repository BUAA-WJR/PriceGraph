#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com

import os
import sys
import json
import pickle
import pyunicorn
import pandas as pd
from pyunicorn import timeseries
from multiprocessing import Pool


PWD = os.path.dirname(os.path.realpath(__file__))


def make_visibility_graph(input_):
    file_, PI, T = input_
    vg_dir = os.path.join('../VG', PI)
    vg_file = os.path.join(vg_dir, '%s.pickle' % file_[:-4])

    df = pd.read_csv(os.path.join('../data', file_), index_col=0)
    df.set_index(df.index.astype('str'), inplace=True)
    vgs = {}
    for i in df.index:
        iloc = list(df.index).index(i)
        if df.loc[:i,:].shape[0] < T:
            continue
        time_series = df.iloc[iloc-T+1:iloc+1][PI]
        if len(set(time_series.values)) == 1:
            continue
        net = timeseries.visibility_graph.VisibilityGraph(time_series.values)
        vgs[i] = net.adjacency
    with open(vg_file, 'wb') as fp:
        pickle.dump(vgs, fp)


if __name__ == '__main__':
    vol_price = ['vol', 'amount', 'high', 'open', 'low', 'close']
    files = os.listdir('../data')
    for PI in vol_price:
        vg_dir = os.path.join('../VG', PI)
        if not os.path.exists(vg_dir):
            os.makedirs(vg_dir)
        pool = Pool()
        pool.map(make_visibility_graph, [(f, PI, 20) for f in files])
        pool.close()
        pool.join()
