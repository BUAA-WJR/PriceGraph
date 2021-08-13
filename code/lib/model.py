#!/usr/bin/env python
# encoding: utf-8
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, time_step, drop_ratio):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # nn.Linear: y=xA+b
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)
        #self.attn = nn.Sequential(attn1, attn2, nn.Tanh(), attn3)
        self.drop_ratio = drop_ratio
        self.drop = nn.Dropout(p=drop_ratio/100.)


    def forward(self, driving_x):
        # driving_x: batch_size, T, input_size
        driving_x = self.drop(driving_x)
        batch_size = driving_x.size(0)
        # batch_size * time_step * hidden_size
        code = self.init_variable(batch_size, self.T, self.hidden_size)
        # initialize hidden state (output)
        h = self.init_variable(1, batch_size, self.hidden_size)
        # initialize cell state
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size)
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            # batch_size * input_size * T
            z1 = self.attn1(x)
            # batch_size * input_size * T
            z2 = self.attn2(driving_x.permute(0, 2, 1))
            # batch_size * input_size * T
            x = z1 + z2
            # batch_size * input_size * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                # batch_size * input_size
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            # batch_size * input_size (element dot multi)
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            h = states[0]  # 1, batch_size, hidden_size
            s = states[1]

            # encoding result
            # batch_size * time_step * encoder_hidden_size
            code[:, t, :] = h

        if self.drop_ratio > 0:
            code = self.drop(code)

        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.attn_ci = nn.Linear(in_features=time_step, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.output = nn.Linear(in_features=code_hidden_size+hidden_size, out_features=hidden_size)

    def forward(self, h, y_seq, cis):
        # h: batch_size * time_step * layer1_hidden_size
        # y_seq: batch_size * time_step
        # cis: batch_size * time_step
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)

        for t in range(self.T):
            # batch_size * time_step * (2 * decoder_hidden_size)
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            # batch_size * time_step * layer1_hidden_size
            z1 = self.attn1(x)
            # batch_size * time_step * layer1_hidden_size
            z2 = self.attn2(h)
            # b * T --> 1 * b * T --> b * T * T --> b * T * layer1_hidden_size
            zci = self.attn_ci(self.embedding_hidden(cis.unsqueeze(0)))
            x = z1 + z2 + zci
            # batch_size * time_step * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                # batch_size * time_step
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.code_hidden_size) + 1
            # batch_size * layer1_hidden_size
            # batch matrix mul: 第一个维度是batch_size，然后剩下的当普通矩阵乘
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)  # (b, 1, T) * (b, T, m)
            # batch_size * (1 + layer1_hidden_size)
            yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
            # batch_size * (1 + layer1_hidden_size)
            y_tilde = self.tilde(yc)
            _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
            d = states[0]  # 1, batch_size, hidden_size
            s = states[1]

        # return d.squeese(0)
        # batch_size * (hidden_size + last_layer_hidden_size)
        dt_ct = torch.cat((d.squeeze(0), ct), dim=1)
        # batch_size * hidden_size
        return self.output(dt_ct)

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)


class DarnnCI(nn.Module):
    def __init__(self, input_size, hidden_size, time_step, drop_ratio):
        super(DarnnCI, self).__init__()
        self.layer1 = AttnEncoder(input_size=input_size, hidden_size=hidden_size, time_step=time_step, drop_ratio=drop_ratio)
        self.layer2 = AttnDecoder(code_hidden_size=hidden_size, hidden_size=hidden_size, time_step=time_step)

    def forward(self, var_x, var_y, var_ci):
        out1 = self.layer1(var_x)
        out2 = self.layer2(out1, var_y, var_ci)
        return out2


class SelfAttention(nn.Module):
    def __init__(self, last_hidden_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.last_hidden_size = last_hidden_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)

    def forward(self, h):
        # h: batch_size * last_hidden_size
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        dk = q.size(-1)
        z = torch.mm(q, k.t()) / math.sqrt(dk)  # (b, hidden_size) * (hidden_size, b) ==> (b, b)
        beta = F.softmax(z, dim=1)
        st = torch.mm(beta, v)  # (b, b) * (b, hidden_size) ==> (b, hidden_size)
        return st


class PriceGraph(nn.Module):
    def __init__(self, input_size, hidden_size, time_step, drop_ratio):
        super(PriceGraph, self).__init__()
        self.das = nn.ModuleList([DarnnCI(input_size, hidden_size, time_step, drop_ratio) for i in range(6)])
        self.attn = SelfAttention(hidden_size, hidden_size)

    def forward(self, var):
        out = 0
        for i in range(6):
            out += self.das[i](var[i]['ems'], var[i]['ys'], var[i]['cis'])
        # batch * hidden_size
        return self.attn(out)


class output_layer(nn.Module):
    def __init__(self, last_hidden_size, output_size=1):
        super(output_layer, self).__init__()
        self.ln = nn.Linear(in_features=last_hidden_size, out_features=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, st):
        #st: b * last_hidden_size
        y_res = self.ln(st)
        # y_res: (batch_size, 1)
        y_res = self.sigmoid(y_res.squeeze(1))
        return y_res  # batch_size


