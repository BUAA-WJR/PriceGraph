# Price graphs: Utilizing the structural information of financial time series for stock prediction
This repository is the official PyTorch implementation of the experiments in the following paper:

Junran Wu, Ke Xu, Xueyuan Chen, Shangzhe Li, Jichang Zhao. Price graphs: Utilizing the structural information of financial time series for stock prediction

[arXiv](https://arxiv.org/abs/2106.02522)

## Installation

Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over PyTorch 1.7.0+cu110 version.

Then install the other dependencies.

```
pip install networkx pyunicorn
```

## Data preparation

* Unfold data

```
tar -xvzf data.tar.gz
```

* Change directory to code

```
cd code
```

* Transform stock prices to graphs

```
python price_graph.py
```

* Calculate collective influence (CI) for graph nodes

```
python price_ci.py
```

* Price graph embedding

```
python price_embedding.py
```

* Generate dataset for train and test

```
python dataset.py
```

## Trainer Usage

Train:

```
usage: trainer.py [-h] [-e EPOCH] [-b BATCH] [-ts TIMESTEP] [-hs HIDDENSIZE] [-y YEARS [YEARS ...]] [-sn SEASON] [-dr DROPRATIO] [-s SPLIT] [-i INTERVAL] [-l LRATE] [-l2 L2RATE] [-t]

Train the price graph model on stock

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCH, --epoch EPOCH
                        the number of epochs
  -b BATCH, --batch BATCH
                        the mini-batch size
  -ts TIMESTEP, --timestep TIMESTEP
                        the length of time_step
  -hs HIDDENSIZE, --hiddensize HIDDENSIZE
                        the length of hidden size
  -y YEARS [YEARS ...], --years YEARS [YEARS ...]
                        an integer for the accumulator
  -sn SEASON, --season SEASON
                        the test season of 2019
  -dr DROPRATIO, --dropratio DROPRATIO
                        the ratio of drop
  -s SPLIT, --split SPLIT
                        the split ratio of validation set
  -i INTERVAL, --interval INTERVAL
                        save models every interval epoch
  -l LRATE, --lrate LRATE
                        learning rate
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda
  -t, --test            train or test
```

An example of training process is as follows:

```
python trainer.py -e 1000 -l 0.001 -hs 32 -ts 20 -b 256 -dr 0 -i 50 -s 30 -l2 0 -y 2017 2018 -sn 1
```

## Base Code Repo

VG algorithm is adopted from [https://github.com/pik-copan/pyunicorn](https://github.com/pik-copan/pyunicorn)

CI algorithm is adopted from [https://github.com/zhfkt/ComplexCi](https://github.com/zhfkt/ComplexCi).

Struc2vec implementation is adopted from [https://github.com/shenweichen/GraphEmbedding](https://github.com/shenweichen/GraphEmbedding).

DARNN implementation in PyTorch is adopted from [https://github.com/ysn2233/attentioned-dual-stage-stock-prediction](https://github.com/ysn2233/attentioned-dual-stage-stock-prediction)



