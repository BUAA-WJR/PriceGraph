#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from ge import Struc2Vec


PWD = os.path.dirname(os.path.realpath(__file__))


def struc2vec_embedding(input_):
    file_, em_size, PI = input_
    vg_file = os.path.join('../VG', PI, file_)
    with open(vg_file, 'rb') as fp:
        vgs = pickle.load(fp)
    em_dir = os.path.join('../Struc2vec' , PI)
    if not os.path.exists(em_dir):
        os.makedirs(em_dir)
    ems = {}
    for d, adj in vgs.items():
        labels = np.array([str(i) for i in range(20)])
        G = nx.Graph()
        for i in range(len(labels)):
            adj_nodes = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(adj_nodes), adj_nodes]))
            G.add_edges_from(edges)
        model = Struc2Vec(G, walk_length=10, num_walks=80, workers=40, verbose=40, stay_prob=0.3, opt1_reduce_len=True, opt2_reduce_sim_calc=True, opt3_num_layers=None, temp_path='./temp_struc2vec_%s/' % PI, reuse=False) #init model
        model.train(embed_size=em_size, window_size=3, workers=40, iter=5)  # train model
        embeddings = model.get_embeddings()  # get embedding vectors
        ems[d] = {k: v.tolist() for k, v in embeddings.items()}
    with open(os.path.join(em_dir, '%s.json' % file_[:-7]), 'w') as fp:
        json.dump(ems, fp)


if __name__ == '__main__':
    vol_price = ['close', 'vol', 'amount', 'high', 'open', 'low']
    Dim = 32
    for PI in vol_price:
        vg_dir = os.path.join('../VG', PI)
        [struc2vec_embedding((f, Dim, PI)) for f in os.listdir(vg_dir)]
