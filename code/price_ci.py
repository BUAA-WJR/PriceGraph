#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com

import os
import sys
import json
import pickle
import numpy as np
import networkx as nx
from multiprocessing import Pool
from lib.ci import collective_influence


def vg_ci(_input):
    file_, PI, T = _input
    graph_file = os.path.join('../VG', PI, file_)
    with open(graph_file, 'rb') as fp:
        vgs = pickle.load(fp)
    cis = {}
    for d, adj in vgs.items():
        labels = np.array([str(i) for i in range(20)])
        G = nx.Graph()
        for i in range(T):
            vg_adjs = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)
        cis[d] = collective_influence(G)
    ci_file = os.path.join('../CI', PI, '%s.json' % file_[:-7])
    with open(ci_file, 'w') as fp:
        json.dump(cis, fp)


if __name__ == '__main__':
    vol_price = ['close', 'vol', 'amount', 'high', 'open', 'low']
    T = 20
    for PI in vol_price:
        vg_dir = os.path.join('../VG/', PI)
        ci_dir = os.path.join('../CI', PI)
        if not os.path.exists(ci_dir):
            os.makedirs(ci_dir)
        pool = Pool()
        pool.map(vg_ci, [(f, PI, T) for f in os.listdir(vg_dir)])
        pool.close()
        pool.join()
